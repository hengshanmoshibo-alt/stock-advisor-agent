from __future__ import annotations

from dataclasses import asdict, dataclass

from .article_loader import Article
from .text_utils import split_paragraphs, term_frequencies


@dataclass
class Chunk:
    chunk_id: str
    article_id: str
    title: str
    published: str
    author: str
    source_path: str
    url: str
    chunk_index: int
    text: str
    preview: str
    term_freq: dict[str, int]
    length: int

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_article(article: Article, chunk_paragraphs: int = 4, overlap: int = 1) -> list[Chunk]:
    paragraphs = split_paragraphs(article.content)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_paragraphs - overlap)
    for index, start in enumerate(range(0, len(paragraphs), step)):
        group = paragraphs[start : start + chunk_paragraphs]
        if not group:
            continue
        text = "\n\n".join(group)
        chunk = Chunk(
            chunk_id=f"{article.article_id}:{index}",
            article_id=article.article_id,
            title=article.title,
            published=article.published,
            author=article.author,
            source_path=article.source_path,
            url=article.url,
            chunk_index=index,
            text=text,
            preview=text[:180],
            term_freq=dict(term_frequencies(f"{article.title}\n{text}")),
            length=max(len(text), 1),
        )
        chunks.append(chunk)
        if start + chunk_paragraphs >= len(paragraphs):
            break
    return chunks


def chunk_articles(articles: list[Article]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article))
    return chunks
