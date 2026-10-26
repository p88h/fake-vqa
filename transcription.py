"""Scoring of transcriptions (OCR text) against the generated text layer.

Both sides are normalized first: markdown-only syntax is removed (the generated
pages contain no `#`, `*`, `|` or backticks, so this cannot hide real errors),
dashes and quotes are unified and all whitespace collapses to single spaces, so
line wrapping and page breaks don't count. Case and punctuation are kept.
Character and word error rates are edits divided by the reference length; they
can exceed 100% when the transcription adds a lot of text.
"""

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

# Our own page separators in text-layer prompts, and markdown horizontal rules
_SEPARATOR_LINE = re.compile(r"(?m)^\s*(-{3,}\s*Page \d+\s*-{3,}|[-*_=]{3,})\s*$")
_TABLE_RULE = re.compile(r"(?m)^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*$")
_CODE_FENCE = re.compile(r"(?m)^\s*```.*$")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
# Markdown escapes, e.g. `3401\.` to stop a line from rendering as a numbered list
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>])")
# Emphasis and code markers wrap words (`**3401**.`), headings and table pipes separate them
_MARKDOWN_CHARS = str.maketrans({"*": "", "`": "", "#": " ", "|": " "})
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for pattern in (_SEPARATOR_LINE, _TABLE_RULE, _CODE_FENCE, _IMAGE, _HTML_TAG):
        text = pattern.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _ESCAPE.sub(r"\1", text)
    text = text.translate(_MARKDOWN_CHARS).translate(_DASHES).translate(_QUOTES)
    return " ".join(text.split())


@dataclass
class EditCounts:
    char_edits: int = 0
    chars: int = 0
    word_edits: int = 0
    words: int = 0

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.char_edits + other.char_edits,
            self.chars + other.chars,
            self.word_edits + other.word_edits,
            self.words + other.words,
        )

    @property
    def cer(self) -> float:
        return self.char_edits / max(self.chars, 1)

    @property
    def wer(self) -> float:
        return self.word_edits / max(self.words, 1)

    def rates(self, prefix: str = "") -> dict:
        return {f"{prefix}cer": round(100 * self.cer, 2), f"{prefix}wer": round(100 * self.wer, 2)}


def compare(reference: str, hypothesis: str) -> EditCounts:
    """Edit counts of `hypothesis` against `reference`, after normalizing both."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    ref_words, hyp_words = ref.split(), hyp.split()
    return EditCounts(
        char_edits=Levenshtein.distance(ref, hyp),
        chars=len(ref),
        word_edits=Levenshtein.distance(ref_words, hyp_words),
        words=len(ref_words),
    )


def total(counts: list[EditCounts]) -> EditCounts:
    return sum(counts, EditCounts())
