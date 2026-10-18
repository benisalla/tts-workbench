"""Text front-end: normalisation and tokenisation.

A TTS model never sees raw strings. The text is first *normalised* (lower-cased, small
numbers spelled out, punctuation reduced to a few symbols) and then *tokenised* into
integer ids: one id per character (`char`) or per phoneme (`phoneme`, via `g2p-en`).
Id 0 is padding, id 1 is the end-of-sentence symbol `~` that every sentence ends with
(it gives autoregressive models a clean signal to stop).
"""

from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """0 <= n < 1_000_000 as English words (enough for dates and counts in captions)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_to_words(n % 100))
    if n < 1_000_000:
        return number_to_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_to_words(n % 1000))
    return str(n)


def normalize(text: str, lowercase: bool = True) -> str:
    """Spell out integers, keep letters, apostrophes, spaces and . , ? ! ; collapse spaces."""
    if lowercase:
        text = text.lower()
    text = re.sub(r"\d+", lambda m: number_to_words(int(m.group())) if len(m.group()) < 7 else m.group(), text)
    text = text.replace("-", " ").replace(";", ",").replace(":", ",")
    text = re.sub(r"[^a-z' .,?!]", " ", text) if lowercase else re.sub(r"[^A-Za-z' .,?!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


PAD, EOS = "_", "~"
CHAR_SYMBOLS = [PAD, EOS, " ", "'", ".", ",", "?", "!"] + list("abcdefghijklmnopqrstuvwxyz")

# ARPAbet (CMUdict) phonemes, as produced by g2p-en, stress digits stripped.
ARPABET = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
           "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y",
           "Z", "ZH"]
PHONEME_SYMBOLS = [PAD, EOS, " ", ".", ",", "?", "!"] + ARPABET


class Tokenizer:
    """Text -> list of ids (and back). `kind` is "char" or "phoneme"."""

    def __init__(self, kind: str = "char", lowercase: bool = True):
        self.kind = kind
        self.lowercase = lowercase
        self.symbols = CHAR_SYMBOLS if kind == "char" else PHONEME_SYMBOLS
        self.index = {s: i for i, s in enumerate(self.symbols)}
        self._g2p = None

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 1

    def _phonemes(self, text: str) -> list[str]:
        if self._g2p is None:
            from g2p_en import G2p  # optional extra: pip install "tts-workbench[phonemes]"

            self._g2p = G2p()
        out = []
        for p in self._g2p(text):
            p = p.rstrip("012")
            if p in self.index:
                out.append(p)
        return out

    def encode(self, text: str) -> list[int]:
        text = normalize(text, self.lowercase)
        units = list(text) if self.kind == "char" else self._phonemes(text)
        ids = [self.index[u] for u in units if u in self.index]
        return ids + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        sep = "" if self.kind == "char" else " "
        return sep.join(self.symbols[i] for i in ids if i > 1)


__all__ = ["CHAR_SYMBOLS", "EOS", "PAD", "PHONEME_SYMBOLS", "Tokenizer", "normalize", "number_to_words"]
