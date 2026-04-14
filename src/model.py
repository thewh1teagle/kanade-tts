"""Autoregressive text-to-Kanade model built from transformers components."""

from __future__ import annotations

from transformers import BertConfig, BertLMHeadModel, BertModel, EncoderDecoderModel


AUDIO_TOKEN_VOCAB_SIZE = 12_800
AUDIO_PAD_TOKEN_ID = AUDIO_TOKEN_VOCAB_SIZE
AUDIO_BOS_TOKEN_ID = AUDIO_TOKEN_VOCAB_SIZE + 1
AUDIO_EOS_TOKEN_ID = AUDIO_TOKEN_VOCAB_SIZE + 2
AUDIO_VOCAB_SIZE = AUDIO_TOKEN_VOCAB_SIZE + 3


def build_model(args, tokenizer) -> EncoderDecoderModel:
    encoder_config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.encoder_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.ffn_dim,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_text_positions,
    )
    decoder_config = BertConfig(
        vocab_size=AUDIO_VOCAB_SIZE,
        pad_token_id=AUDIO_PAD_TOKEN_ID,
        bos_token_id=AUDIO_BOS_TOKEN_ID,
        eos_token_id=AUDIO_EOS_TOKEN_ID,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.decoder_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.ffn_dim,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_audio_positions,
        is_decoder=True,
        add_cross_attention=True,
    )

    model = EncoderDecoderModel(
        encoder=BertModel(encoder_config),
        decoder=BertLMHeadModel(decoder_config),
    )
    model.config.pad_token_id = AUDIO_PAD_TOKEN_ID
    model.config.decoder_start_token_id = AUDIO_BOS_TOKEN_ID
    model.config.eos_token_id = AUDIO_EOS_TOKEN_ID
    model.config.vocab_size = AUDIO_VOCAB_SIZE
    model.generation_config.pad_token_id = AUDIO_PAD_TOKEN_ID
    model.generation_config.decoder_start_token_id = AUDIO_BOS_TOKEN_ID
    model.generation_config.eos_token_id = AUDIO_EOS_TOKEN_ID
    return model
