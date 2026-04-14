"""Character-level tokenizer for IPA phoneme sequences.

Vocab:
  - Special tokens: [PAD], [BOS], [EOS], [UNK]
  - IPA phonemes fetched from piper-checkpoints config

No normalizer — IPA input is already clean.
"""

from __future__ import annotations

import string
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from tokenizers import Regex
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

# IPA symbols sourced from:
# https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ljspeech/medium/config.json
_ASCII = list(string.ascii_lowercase + "X" + string.digits + ' !"#$\'(),-.:;?^_' + " ")
_IPA_EXTRAS = [
    "æ", "ç", "ð", "ø", "ħ", "ŋ", "œ",
    "ǀ", "ǁ", "ǂ", "ǃ",
    "ɐ", "ɑ", "ɒ", "ɓ", "ɔ", "ɕ", "ɖ", "ɗ", "ɘ", "ə", "ɚ", "ɛ", "ɜ", "ɞ",
    "ɟ", "ɠ", "ɡ", "ɢ", "ɣ", "ɤ", "ɥ", "ɦ", "ɧ", "ɨ", "ɪ", "ɫ", "ɬ", "ɭ",
    "ɮ", "ɯ", "ɰ", "ɱ", "ɲ", "ɳ", "ɴ", "ɵ", "ɶ", "ɸ", "ɹ", "ɺ", "ɻ", "ɽ",
    "ɾ", "ʀ", "ʁ", "ʂ", "ʃ", "ʄ", "ʈ", "ʉ", "ʊ", "ʋ", "ʌ", "ʍ", "ʎ", "ʏ",
    "ʐ", "ʑ", "ʒ", "ʔ", "ʕ", "ʘ", "ʙ", "ʛ", "ʜ", "ʝ", "ʟ", "ʡ", "ʢ", "ʦ",
    "ʰ", "ʲ", "ˈ", "ˌ", "ː", "ˑ", "˞", "ˤ",
    "̃", "̧", "̩", "̪", "̯", "̺", "̻",
    "β", "ε", "θ", "χ", "ᵻ", "↑", "↓", "ⱱ",
]
IPA_PHONEMES = sorted(set(_ASCII + _IPA_EXTRAS), key=ord)


def build_vocab() -> dict[str, int]:
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for p in IPA_PHONEMES:
        if p not in vocab:
            vocab[p] = len(vocab)
    return vocab


def build_tokenizer() -> Tokenizer:
    vocab = build_vocab()

    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]", continuing_subword_prefix="##"))
    tokenizer.pre_tokenizer = Split(pattern=Regex("[\\s\\S]"), behavior="isolated")

    bos_id = vocab["[BOS]"]
    eos_id = vocab["[EOS]"]
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in SPECIAL_TOKENS])

    return tokenizer


def save_tokenizer(path: str | Path) -> None:
    build_tokenizer().save(str(path))


@lru_cache(maxsize=None)
def load_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=build_tokenizer(),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
