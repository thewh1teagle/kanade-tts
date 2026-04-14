# Hebrew TTS Architecture Summary

## Problem

Build a non-autoregressive Hebrew TTS model that:
- Takes unvowelized Hebrew text as input
- Produces natural speech output
- Is fast at inference (single forward pass, all frames in parallel)
- Leverages the existing renikud G2P model as a front-end
- Supports zero-shot voice cloning from a short reference clip

---

## Ideas Explored

### Codec Choice

**SNAC** — hierarchical RVQ codec with 3 codebook levels (coarse → mid → fine). Good quality but complex: predicting 3 token sequences simultaneously requires coupled heads or hierarchical decoding, and the multi-level structure adds training complexity.

**Kanade Tokenizer** (`frothywater/kanade-tokenizer`) — single-layer disentangled speech tokenizer:
- Vocabulary: 12,800 tokens
- Token rate: 12.5 or 25 Hz
- Two outputs: discrete content tokens + global speaker embedding
- Single flat token sequence — no hierarchical codebooks
- Speaker identity is naturally disentangled into a continuous embedding

**Decision: Kanade.** Single token sequence means straightforward classification heads, no hierarchical complexity. Speaker embedding comes for free — no separate style encoder needed.

---

### Duration / Alignment

Several approaches considered:

**Forced alignment (MFA/WhisperX)** — extract per-phoneme timestamps offline. Hebrew support is limited and adds data prep complexity.

**Monotonic Alignment Search (MAS)** — learns alignment end-to-end (StyleTTS2/Glow-TTS style). No preprocessing but complex to implement.

**BlueTTS duration predictor** — predicts total utterance duration (single scalar). Simple, no labels needed beyond audio length.

**Stylish-TTS duration predictor** — predicts per-phoneme durations using AdaptiveConvNeXtBlocks conditioned on a style vector. Better prosody control than total-duration approach. Uses a differentiable cumsum trick for soft alignment — no MFA labels needed.

**Decision: Stylish-TTS style per-phoneme duration predictor**, conditioned on Kanade speaker embedding via AdaptiveLayerNorm. Alignment (how many frames per phoneme) is learned implicitly from training data — no forced aligner needed.

---

### Style / Speaker Conditioning

**Separate style encoder** (BlueTTS/Stylish-TTS) — encodes reference audio into a style vector. Rejected because Kanade already outputs a global speaker embedding for free.

**Kanade speaker embedding** — injected via AdaptiveLayerNorm into every conditioning point (duration predictor and token heads). Zero-shot VC works automatically because the model conditions on the embedding, not a speaker ID.

**AdaptiveLayerNorm** (from Stylish-TTS) — replaces standard LayerNorm. Projects the style vector into scale (gamma) and shift (beta) that modulate normalized activations:
```python
x = LayerNorm(x)
x = (1 + gamma) * x + beta   # gamma, beta predicted from speaker embedding
```
Low parameter cost, deep style integration at every layer.

---

### Acoustic Generation

**Flow matching** — iterative denoising, best quality ceiling, multiple forward passes at inference.

**Masked prediction** — iterative unmasking of discrete tokens, good parallelism.

**Direct classification (renikud-style)** — predict discrete tokens via coupled classification heads. Single forward pass, all frames in parallel, architecturally simple.

**Decision: Direct classification.** Architecturally consistent with renikud, fastest inference. Flow matching can replace the heads later if quality ceiling is insufficient.

---

## Final Architecture

```
[Input]
unvowelized Hebrew text
        ↓
   renikud (G2P)
        ↓
   IPA phoneme sequence

[Text Encoder]
ModernBERT-style encoder (same family as renikud)
        ↓
   per-phoneme hidden states

[Duration Predictor]
AdaptiveConvNeXtBlocks conditioned on Kanade speaker embedding
        ↓
   per-phoneme frame counts
        ↓
   length regulator → N frame hidden states

[Token Prediction Heads]  (coupled)
frame hidden states (AdaptiveLayerNorm on Kanade speaker embedding)
        ↓
   head_1 → Kanade token logits
   head_2 (hidden + head_1 logits) → Kanade token logits

[Output]
Kanade token sequence → Kanade decoder → waveform

[Speaker Embedding]
reference audio clip → Kanade encoder → speaker embedding
                                              ↓
                              injected via AdaptiveLayerNorm
                              into duration predictor AND token heads
```

---

## How Alignment is Learned

No explicit alignment supervision needed:
- **Duration predictor** learns phoneme→frame-count from the actual Kanade token sequence lengths in training data
- **Token heads** learn what token to emit at each frame position via teacher forcing on ground truth token sequences
- Both are learned end-to-end from `(IPA sequence, Kanade token sequence)` pairs only

---

## Inference Speed

Single forward pass — all N frames predicted in parallel:

| Model type | Steps for 5s clip (~125 tokens) |
|---|---|
| Autoregressive (VALL-E) | 125 forward passes |
| Flow matching (F5-TTS) | 16–32 forward passes |
| This model | 1 forward pass |

Bottleneck at inference is the Kanade decoder, not the TTS model.

---

## Zero-Shot Voice Cloning

Free by design — the model never sees speaker IDs, only Kanade speaker embeddings. At inference, encode any reference clip through Kanade's encoder and inject the embedding. The model generalizes to unseen voices because it learned to condition on the embedding space, not memorize specific speakers.

---

## Data Pipeline

```
raw audio → Kanade encoder → (token sequences, speaker embeddings)  [offline, once]
raw text  → renikud → IPA phoneme sequences

training pairs: (IPA sequence, Kanade token sequence, speaker embedding)
```

No forced alignment. No phoneme timestamps. Duration supervision comes from token sequence length.

---

## Key Properties

| Property | Value |
|---|---|
| Input | Unvowelized Hebrew text |
| G2P front-end | renikud |
| Acoustic token vocab | 12,800 (Kanade) |
| Token rate | 12.5 or 25 Hz |
| Duration | Per-phoneme, learned from data (no MFA) |
| Style conditioning | Kanade speaker embedding via AdaptiveLayerNorm |
| Alignment | Explicit length regulator, durations learned end-to-end |
| Generation | Single forward pass, all frames in parallel |
| Zero-shot VC | Yes, by design |
| Inference speed | Very fast — no iterative denoising |

---

## Future Improvements

- Replace classification heads with flow matching for better prosody variation
- Replace length regulator with cross-attention alignment (BlueTTS style) for more natural rhythm
- Multi-speaker training with more data
