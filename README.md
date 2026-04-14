# kanade-tts

Small autoregressive Hebrew TTS that maps `ipa_tokens -> Kanade audio_tokens`.

## Train

```bash
bash scripts/train_scratch.sh
```

## Infer

```bash
bash scripts/train_infer.sh outputs/kanade-ar/step-8000
```

## Architecture

- `BertModel` text encoder
- `BertLMHeadModel` causal decoder with cross-attention
- `EncoderDecoderModel` wrapper
- Kanade decoder + vocoder for waveform synthesis

See [docs/ARCHITECTURE.md](/home/yakov/Documents/audio/kanade-tts/docs/ARCHITECTURE.md).
