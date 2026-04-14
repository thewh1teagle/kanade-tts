# Architecture

Autoregressive Hebrew TTS model that predicts Kanade acoustic tokens from IPA text.

## Overview

```text
IPA text / ipa_tokens
        ↓
small bidirectional text encoder (BERT-style)
        ↓
causal decoder with cross-attention to text
        ↓
next Kanade token prediction
        ↓
Kanade decoder + vocoder
        ↓
waveform
```

## Design

- Training target is the existing `audio_tokens` field in `dataset/dataset.jsonl`.
- Decoder input is teacher-forced: `[BOS] + audio_tokens[:-1]`.
- Labels are `audio_tokens + [EOS]`.
- Loss is plain cross-entropy over the Kanade token vocabulary plus special tokens.
- The model is built mostly from `transformers` components:
  - `BertModel` encoder
  - `BertLMHeadModel` decoder with `is_decoder=True` and `add_cross_attention=True`
  - wrapped by `EncoderDecoderModel`

## Why This Replaced The Previous Design

- No MAS alignment branch.
- No duration predictor.
- No length regulator.
- No per-frame independent token heads.
- No speaker-conditioning path in the TTS model itself.

This is a better fit for the current dataset because it uses direct supervised text-to-token training and avoids alignment machinery that was not learning correctly.

## Model Size

Default config targets roughly the mid-20M range:

- hidden size: `320`
- encoder layers: `6`
- decoder layers: `8`
- attention heads: `5`
- FFN dim: `1280`

Adjust these in `src/config.py` if needed.

## Dataset Contract

Each training sample currently needs:

```json
{
  "ipa_tokens": [...],
  "audio_tokens": [...]
}
```

`speaker_embedding` may remain in the JSONL for downstream waveform decoding, but it is not consumed by the AR TTS model.
