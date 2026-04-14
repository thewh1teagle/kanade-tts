# Flow Matching Replacement for Token Heads

## Problem

The current classification heads predict 1-of-12,800 audio tokens per frame from a 512-dim vector. After 3500+ training steps, token accuracy is 1.1% and cross-entropy loss is stuck at ~17.6 (random chance for two heads summed is ~18.9). The model learned token frequency bias but nothing useful.

Root causes:
- 12,800-way classification from a single vector per frame is too hard
- No frame-level context — each frame is predicted independently from a repeated phoneme vector
- MAS alignment uses randomly initialized `audio_embed`, giving garbage signal early in training (chicken-and-egg)

## Solution

Replace the two classification heads with a **conditional flow matching** model that predicts continuous Kanade codebook embeddings (768-dim) instead of discrete token IDs. At inference, quantize via nearest-neighbor lookup into the frozen codebook.

---

## What Changes

### Remove
- `model.audio_embed` — `nn.Embedding(12800, 512)` (learned, random init)
- `model.audio_proj` — `nn.Linear(512, 512)`
- `model.heads` — `TokenHeads` (two coupled classification heads)
- `src/modules/heads.py` — entire file

### Add

#### `src/modules/flow.py` — Flow matching velocity network
- 6x ConvNeXt blocks (reuse `AdaptiveConvNeXtBlock` pattern from duration.py)
- Conditioned on speaker embedding via AdaptiveLayerNorm (same as duration predictor)
- Timestep conditioning via sinusoidal embedding projected to channel dim, added as bias
- Input per frame: `cat(noised_target_768, conditioning_512)` → project to 512 channels
- Output per frame: predicted velocity (768-dim)
- ~5M params

#### `scripts/extract_codebook.py` — One-time codebook extraction
- Load Kanade model, enumerate all 12,800 FSQ indices
- Map through `fsq.indices_to_codes()` → `proj_out()` to get (12800, 768) embeddings
- Save as `dataset/codebook.pt`
- This avoids loading the full Kanade model during training

### Modify

#### `src/model.py` — Core model changes
```
Before:
  expanded_hidden (B, T, 512) → TokenHeads → 12800 logits × 2 → cross_entropy

After:
  expanded_hidden (B, T, 512) → FlowModel(x_t, t, cond, spk) → velocity → MSE loss
```

Specific changes:
1. Load frozen codebook buffer `(12800, 768)` from `dataset/codebook.pt`
2. MAS alignment: look up `codebook[audio_tokens]` for target embeddings (frozen, meaningful from step 0) instead of learned `audio_embed`. Project 768→512 for similarity with encoder hidden states.
3. Training forward pass:
   - Target: `x_1 = codebook[audio_tokens]`  — shape `(B, T_frame, 768)`
   - Sample `t ~ U(0,1)` per batch element
   - Sample noise `x_0 ~ N(0,1)` — shape `(B, T_frame, 768)`
   - Interpolate `x_t = (1-t)*x_0 + t*x_1`
   - Velocity target: `v = x_1 - x_0`
   - Predict: `v_pred = flow_model(x_t, t, expanded_hidden, speaker_emb)`
   - Loss: `MSE(v_pred, v)` masked to valid frames
4. Return `flow_loss + dur_loss`

#### `src/eval.py` — Updated evaluation
- Compute flow loss (MSE) on validation set
- Compute token accuracy via nearest-neighbor into codebook (only at eval, not training)
- Run a few-step ODE to get predicted embeddings, then NN lookup

#### `src/infer.py` — ODE sampling at inference
```python
# Replace: logits → argmax
# With:
x = torch.randn(1, T_frame, 768)       # start from noise
for i in range(num_steps):              # 8 Euler steps
    t = i / num_steps
    t_tensor = torch.full((1,), t)
    v = model.flow_model(x, t_tensor, expanded_hidden, speaker_emb)
    x = x + v / num_steps
audio_tokens = cdist(x, codebook).argmin(dim=-1)  # nearest neighbor
```

#### `src/metrics.py` — Log flow_loss instead of tok_loss

#### `src/modules/__init__.py` — Export `FlowModel` instead of `TokenHeads`

#### `src/config.py` — Add args
- `--codebook-path` (default: `dataset/codebook.pt`)
- `--flow-layers` (default: 6)
- `--flow-channels` (default: 512)
- `--inference-steps` (default: 8)

---

## Why This Works

1. **Continuous target is easier** — predicting a 768-dim vector (MSE) is dramatically easier than 12,800-way classification. The embedding space is smooth: nearby embeddings sound similar, so "close but wrong" still produces reasonable audio.

2. **MAS gets real signal from step 0** — the frozen Kanade codebook embeddings are pretrained and meaningful. The similarity matrix between encoder hidden states and codebook embeddings will produce useful alignments immediately, breaking the chicken-and-egg problem.

3. **Frame-level context via ConvNeXt** — the flow model's depthwise convolutions let each frame attend to neighbors, solving the "independent prediction" problem.

4. **Inference: 1 pass → ~8 passes** — still 15x faster than autoregressive. The Kanade decoder remains the bottleneck.

---

## Execution Order

1. `scripts/extract_codebook.py` — extract and save codebook
2. `src/modules/flow.py` — new flow model
3. `src/model.py` — swap heads for flow, update MAS
4. `src/modules/__init__.py` — update exports
5. `src/config.py` — add new args
6. `src/eval.py` — update metrics
7. `src/metrics.py` — update logging
8. `src/infer.py` — ODE sampling
9. `src/modules/heads.py` — delete or keep as dead code
10. `docs/ARCHITECTURE.md` — update diagram

---

## Parameter Count Impact

| Component | Before | After |
|---|---|---|
| Text Encoder | ~16M | ~16M (unchanged) |
| Duration Predictor | ~3M | ~3M (unchanged) |
| Token Heads | ~20M | — (removed) |
| Audio Embed + Proj | ~6.5M | — (removed) |
| Flow Model | — | ~5M (new) |
| Codebook Proj (768→512) | — | ~0.4M (new) |
| Codebook buffer (frozen) | — | ~9.8M (not trained) |
| **Total trainable** | **~45M** | **~24M** |
