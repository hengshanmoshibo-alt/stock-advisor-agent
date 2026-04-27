from __future__ import annotations

import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from .cleaning import clean_author, clean_content, clean_paragraph, clean_title, repair_mojibake


CONTENT_BLOCK_TAGS = {"p", "section", "blockquote", "li"}


@dataclass
class Article:
    article_id: str
    title: str
    published: str
    author: str
    source_path: str
    url: str
    content: str


class WechatArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.author_parts: list[str] = []
        self.paragraphs: list[str] = []
        self.current_block_parts: list[str] = []
        self.fallback_content_parts: list[str] = []
        self._capture_title = False
        self._capture_author = False
        self._content_depth = 0
        self._block_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        tag_id = attr_map.get("id", "")
        tag_class = attr_map.get("class", "")

        if tag == "span" and "js_title_inner" in tag_class:
            self._capture_title = True
        elif tag == "h1" and tag_id == "activity-name":
            self._capture_title = True

        if tag == "a" and tag_id == "js_name":
            self._capture_author = True

        if tag_id == "js_content":
            self._content_depth = 1
            return

        if self._content_depth > 0:
            self._content_depth += 1
            if tag in CONTENT_BLOCK_TAGS:
                if self._block_depth == 0:
                    self.current_block_parts = []
                self._block_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._capture_title and tag in {"span", "h1"}:
            self._capture_title = False

        if self._capture_author and tag == "a":
            self._capture_author = False

        if self._content_depth > 0:
            if tag in CONTENT_BLOCK_TAGS and self._block_depth > 0:
                self._block_depth -= 1
                if self._block_depth == 0:
                    paragraph = clean_paragraph(" ".join(self.current_block_parts))
                    if paragraph:
                        self.paragraphs.append(paragraph)
                    self.current_block_parts = []
            self._content_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self.title_parts.append(data)
        if self._capture_author:
            self.author_parts.append(data)
        if self._content_depth > 0:
            self.fallback_content_parts.append(data)
            if self._block_depth > 0:
                self.current_block_parts.append(data)

    def as_tuple(self) -> tuple[str, str, str]:
        title = clean_title(" ".join(self.title_parts))
        author = clean_author(" ".join(self.author_parts))
        if self.paragraphs:
            content = clean_content("\n\n".join(self.paragraphs))
        else:
            content = clean_content(" ".join(self.fallback_content_parts))
        return title, author, content


def _article_content_from_html(raw_html: str) -> tuple[str, str, str]:
    parser = WechatArticleParser()
    parser.feed(raw_html)
    parser.close()
    return parser.as_tuple()


def _resolve_source_path(article_dir: Path, output_path: str) -> Path:
    repaired_output_path = repair_mojibake(output_path)
    candidates = [
        Path(repaired_output_path),
        article_dir / Path(repaired_output_path).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    file_name = Path(repaired_output_path).name
    prefix = file_name.split("_", 1)[0]
    if prefix:
        matches = sorted(article_dir.glob(f"{prefix}_*.htm*"))
        if len(matches) == 1:
            return matches[0]

    raise FileNotFoundError(f"Could not locate article HTML for: {output_path}")


def load_articles(article_dir: Path) -> list[Article]:
    manifest_path = article_dir / "_manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))

    articles: list[Article] = []
    for entry in entries:
        source_path = _resolve_source_path(article_dir, entry["output_path"])
        raw_html = source_path.read_text(encoding="utf-8", errors="ignore")
        title, author, content = _article_content_from_html(raw_html)
        articles.append(
            Article(
                article_id=entry["aid"],
                title=title or clean_title(entry["title"]),
                published=entry["published"],
                author=author or "投资数字人资料库",
                source_path=str(source_path),
                url=entry["url"],
                content=content,
            )
        )

    return articles
