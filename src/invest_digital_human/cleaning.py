from __future__ import annotations

import re
import unicodedata
from typing import Iterable


SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
NOISE_PATTERNS = [
    re.compile(r"^微信扫一扫关注.*$"),
    re.compile(r"^继续滑动看下一个.*$"),
    re.compile(r"^喜欢此内容的人还喜欢.*$"),
    re.compile(r"^预览时标签不可点.*$"),
    re.compile(r"^分享收藏点赞在看.*$"),
    re.compile(r"^点击.*(阅读原文|关注我们).*$"),
]
MOJIBAKE_HINTS = "æçéèêåïîðñãâ€œâ€\u20ac鍏閲鎴恄鏂囩珷"


def normalize_display_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ").replace("\ufeff", "")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def repair_mojibake(text: str) -> str:
    original = normalize_display_text(text)
    if not original:
        return ""

    candidates = [original]
    for source_encoding in ("gb18030", "gbk", "latin-1", "cp1252"):
        try:
            repaired = original.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        candidates.append(normalize_display_text(repaired))

    return min(candidates, key=_quality_score)


def clean_title(text: str) -> str:
    return repair_mojibake(text)


def clean_author(text: str) -> str:
    return repair_mojibake(text)


def clean_content(text: str) -> str:
    text = repair_mojibake(text)
    paragraphs = [clean_paragraph(paragraph) for paragraph in text.split("\n\n")]
    filtered = _dedupe_consecutive(paragraph for paragraph in paragraphs if paragraph)
    return MULTI_BLANK_RE.sub("\n\n", "\n\n".join(filtered)).strip()


def clean_paragraph(text: str) -> str:
    paragraph = normalize_display_text(repair_mojibake(text))
    if not paragraph:
        return ""
    if _looks_like_noise(paragraph):
        return ""
    return paragraph


def _looks_like_noise(text: str) -> bool:
    if len(text) <= 1:
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _dedupe_consecutive(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    previous = None
    for value in values:
        if value and value != previous:
            deduped.append(value)
        previous = value
    return deduped


def _quality_score(text: str) -> float:
    if not text:
        return 1e9
    mojibake_penalty = sum(text.count(char) for char in MOJIBAKE_HINTS)
    cjk_bonus = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    control_penalty = sum(1 for char in text if ord(char) < 32 and char not in "\n\t")
    return mojibake_penalty * 8 + control_penalty * 20 - cjk_bonus * 0.05
