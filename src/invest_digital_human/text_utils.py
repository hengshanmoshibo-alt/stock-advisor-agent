from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")
ASCII_RE = re.compile(r"[a-z0-9][a-z0-9.+_-]*")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip().lower()


def split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for raw in re.split(r"\n{2,}", text):
        parts = []
        for line in raw.splitlines():
            normalized_line = normalize_text(line)
            if normalized_line:
                parts.append(normalized_line)
        if parts:
            paragraphs.append(" ".join(parts))
    return paragraphs


def _chinese_terms(text: str) -> list[str]:
    terms: list[str] = []
    for block in CHINESE_RE.findall(text):
        if len(block) <= 4:
            terms.append(block)
        else:
            terms.extend(block[i : i + 2] for i in range(len(block) - 1))
            terms.extend(block[i : i + 3] for i in range(len(block) - 2))
    return terms


def _ascii_terms(text: str) -> list[str]:
    return ASCII_RE.findall(text)


def extract_terms(text: str) -> list[str]:
    normalized = normalize_text(text)
    return _ascii_terms(normalized) + _chinese_terms(normalized)


def term_frequencies(text: str) -> Counter[str]:
    return Counter(extract_terms(text))


def snippet(text: str, limit: int = 160) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def recency_boost(date_text: str) -> float:
    if not date_text:
        return 0.0
    year = int(date_text[:4])
    month = int(date_text[5:7])
    return ((year - 2021) * 12 + month) / 100.0


def safe_log(value: float) -> float:
    return math.log(max(value, 1e-9))
