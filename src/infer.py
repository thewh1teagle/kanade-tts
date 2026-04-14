"""Run inference with a trained autoregressive Kanade TTS checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode
from transformers import EncoderDecoderModel

sys.path.insert(0, str(Path(__file__).parent))

from model import AUDIO_BOS_TOKEN_ID, AUDIO_EOS_TOKEN_ID, AUDIO_PAD_TOKEN_ID
from tokenization import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True, help="Reference audio for Kanade global embedding")
    parser.add_argument("--ipa", type=str, required=True)
    parser.add_argument("--output", type=str, default="/tmp/kanade_out.wav")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def strip_special_tokens(tokens: torch.Tensor) -> torch.Tensor:
    keep = (tokens >= 0) & (tokens < AUDIO_PAD_TOKEN_ID)
    return tokens[keep]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Kanade model...")
    kanade = KanadeModel.from_pretrained("frothywater/kanade-25hz-clean").eval().to(device)
    vocoder = load_vocoder(kanade.config.vocoder_name).to(device)

    print("Loading AR TTS model...")
    tokenizer = load_tokenizer()
    model = EncoderDecoderModel.from_pretrained(args.checkpoint).eval().to(device)
    model.generation_config.decoder_start_token_id = AUDIO_BOS_TOKEN_ID
    model.generation_config.eos_token_id = AUDIO_EOS_TOKEN_ID
    model.generation_config.pad_token_id = AUDIO_PAD_TOKEN_ID

    print("Extracting Kanade global embedding from reference...")
    ref_audio = load_audio(args.reference, sample_rate=kanade.config.sample_rate).to(device)
    with torch.inference_mode():
        ref_features = kanade.encode(ref_audio, return_content=False)
    speaker_embedding = ref_features.global_embedding.squeeze(0)

    print(f"Synthesizing: {args.ipa}")
    input_ids = torch.tensor([tokenizer.encode(args.ipa, add_special_tokens=True)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        audio_tokens = strip_special_tokens(generated[0])
        mel = kanade.decode(
            content_token_indices=audio_tokens,
            global_embedding=speaker_embedding,
        )
        wav = vocode(vocoder, mel.unsqueeze(0))

    sf.write(args.output, wav.squeeze().cpu().numpy(), kanade.config.sample_rate)
    print(f"Generated {audio_tokens.numel()} Kanade tokens")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
