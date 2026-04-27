from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .article_loader import load_articles
from .chunking import Chunk, chunk_articles
from .embeddings import DenseEmbedder
from .text_utils import extract_terms, normalize_text, recency_boost, safe_log, snippet

VECTOR_DIM = 2048
SEARCH_MODES = {"sparse", "vector", "hybrid"}


@dataclass
class SearchHit:
    score: float
    chunk: Chunk
    bm25_score: float
    vector_score: float
    rerank_score: float | None = None


class RAGIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        doc_freq: dict[str, int],
        avg_doc_len: float,
        vectors: np.ndarray | None = None,
        vector_dim: int = VECTOR_DIM,
        vector_backend: str = "hash",
        vector_model: str | None = None,
    ):
        self.chunks = chunks
        self.doc_freq = doc_freq
        self.avg_doc_len = avg_doc_len or 1.0
        self.total_docs = len(chunks)
        self.vector_dim = vector_dim
        self.vector_backend = vector_backend
        self.vector_model = vector_model
        if vectors is None:
            self.vectors = np.zeros((len(chunks), vector_dim), dtype=np.float32)
        else:
            self.vectors = vectors.astype(np.float32, copy=False)

    @classmethod
    def build(
        cls,
        article_dir: Path,
        *,
        dense_embedder: DenseEmbedder | None = None,
    ) -> "RAGIndex":
        articles = load_articles(article_dir)
        chunks = chunk_articles(articles)
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            doc_freq.update(set(chunk.term_freq))
        doc_freq_dict = dict(doc_freq)
        avg_doc_len = sum(chunk.length for chunk in chunks) / max(len(chunks), 1)

        if dense_embedder is not None and chunks:
            texts = [f"{chunk.title}\n{chunk.text}" for chunk in chunks]
            vectors = dense_embedder.embed_texts(texts)
            vector_dim = int(vectors.shape[1])
            vector_backend = dense_embedder.backend
            vector_model = dense_embedder.model_name
        else:
            vectors = cls._build_hash_vectors(
                chunks=chunks,
                doc_freq=doc_freq_dict,
                total_docs=len(chunks),
                vector_dim=VECTOR_DIM,
            )
            vector_dim = VECTOR_DIM
            vector_backend = "hash"
            vector_model = None

        return cls(
            chunks=chunks,
            doc_freq=doc_freq_dict,
            avg_doc_len=avg_doc_len,
            vectors=vectors,
            vector_dim=vector_dim,
            vector_backend=vector_backend,
            vector_model=vector_model,
        )

    @classmethod
    def load(cls, index_path: Path) -> "RAGIndex":
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload["chunks"]]
        vector_dim = payload.get("vector_dim", VECTOR_DIM)
        vectors = None
        vector_path = cls._vector_sidecar_path(index_path)
        if vector_path.exists():
            vectors = np.load(vector_path).astype(np.float32, copy=False)
        return cls(
            chunks=chunks,
            doc_freq=payload["doc_freq"],
            avg_doc_len=payload["avg_doc_len"],
            vectors=vectors,
            vector_dim=vector_dim,
            vector_backend=payload.get("vector_backend", "hash"),
            vector_model=payload.get("vector_model"),
        )

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_freq": self.doc_freq,
            "avg_doc_len": self.avg_doc_len,
            "vector_dim": self.vector_dim,
            "vector_backend": self.vector_backend,
            "vector_model": self.vector_model,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(
            self._vector_sidecar_path(index_path),
            self.vectors.astype(np.float16, copy=False),
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        mode: str = "hybrid",
        *,
        dense_embedder: DenseEmbedder | None = None,
    ) -> list[SearchHit]:
        if mode not in SEARCH_MODES:
            raise ValueError(f"Unsupported search mode: {mode}")

        query_terms = extract_terms(query)
        if not query_terms:
            return []

        query_counter = Counter(query_terms)
        bm25_scores = (
            [self._bm25_score(query_counter, chunk) for chunk in self.chunks]
            if mode in {"sparse", "hybrid"}
            else [0.0] * len(self.chunks)
        )
        query_vector = self._build_query_vector(
            query=query,
            query_counter=query_counter,
            dense_embedder=dense_embedder,
            use_vector=mode in {"vector", "hybrid"},
        )
        vector_scores = (
            (self.vectors @ query_vector).tolist()
            if query_vector.any() and len(self.vectors) == len(self.chunks)
            else [0.0] * len(self.chunks)
        )

        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        max_vector = max(vector_scores) if vector_scores else 0.0
        normalized_query = normalize_text(query)

        hits: list[SearchHit] = []
        for index, chunk in enumerate(self.chunks):
            bm25 = bm25_scores[index]
            vector_score = max(0.0, float(vector_scores[index]))
            if bm25 <= 0 and vector_score <= 0:
                continue

            title_text = normalize_text(chunk.title)
            title_bonus = 0.0
            exact_bonus = 0.0
            if normalized_query and normalized_query in title_text:
                exact_bonus += 0.35
            for term in query_counter:
                if term in title_text:
                    title_bonus += 0.08
                if term in chunk.text:
                    exact_bonus += 0.01

            sparse_component = (bm25 / max_bm25) if max_bm25 > 0 else 0.0
            vector_component = (vector_score / max_vector) if max_vector > 0 else 0.0
            if mode == "sparse":
                score = sparse_component
            elif mode == "vector":
                score = vector_component
            else:
                score = 0.60 * sparse_component + 0.40 * vector_component
            score += title_bonus + exact_bonus + recency_boost(chunk.published)

            hits.append(
                SearchHit(
                    score=score,
                    chunk=chunk,
                    bm25_score=bm25,
                    vector_score=vector_score,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def stats(self) -> dict[str, int | str | None]:
        articles = {chunk.article_id for chunk in self.chunks}
        return {
            "articles": len(articles),
            "chunks": len(self.chunks),
            "vector_dim": self.vector_dim,
            "vector_backend": self.vector_backend,
            "vector_model": self.vector_model,
        }

    @staticmethod
    def _vector_sidecar_path(index_path: Path) -> Path:
        return index_path.with_suffix(".vectors.npy")

    def _build_query_vector(
        self,
        *,
        query: str,
        query_counter: Counter[str],
        dense_embedder: DenseEmbedder | None,
        use_vector: bool,
    ) -> np.ndarray:
        if not use_vector:
            return np.zeros(self.vector_dim, dtype=np.float32)

        if self.vector_backend == "hash":
            return self._vectorize_counter(query_counter)

        if dense_embedder is None:
            return np.zeros(self.vector_dim, dtype=np.float32)

        query_vector = dense_embedder.embed_query(query)
        if query_vector.shape[0] != self.vector_dim:
            raise ValueError(
                f"Embedding dimension mismatch: index={self.vector_dim}, query={query_vector.shape[0]}"
            )
        return query_vector.astype(np.float32, copy=False)

    def _bm25_score(self, query_counter: Counter[str], chunk: Chunk) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        for term, qf in query_counter.items():
            tf = chunk.term_freq.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = safe_log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + k1 * (1 - b + b * (chunk.length / self.avg_doc_len))
            score += idf * ((tf * (k1 + 1)) / denom) * qf
        return score

    @staticmethod
    def _build_hash_vectors(
        chunks: list[Chunk],
        doc_freq: dict[str, int],
        total_docs: int,
        vector_dim: int,
    ) -> np.ndarray:
        vectors = np.zeros((len(chunks), vector_dim), dtype=np.float32)
        for row, chunk in enumerate(chunks):
            vectors[row] = RAGIndex._vectorize_term_freq(
                term_freq=chunk.term_freq,
                doc_freq=doc_freq,
                total_docs=total_docs,
                vector_dim=vector_dim,
            )
        return vectors

    def _vectorize_counter(self, query_counter: Counter[str]) -> np.ndarray:
        return self._vectorize_term_freq(
            term_freq=dict(query_counter),
            doc_freq=self.doc_freq,
            total_docs=self.total_docs,
            vector_dim=self.vector_dim,
        )

    @staticmethod
    def _vectorize_term_freq(
        term_freq: dict[str, int],
        doc_freq: dict[str, int],
        total_docs: int,
        vector_dim: int,
    ) -> np.ndarray:
        vector = np.zeros(vector_dim, dtype=np.float32)
        for term, tf in term_freq.items():
            if tf <= 0:
                continue
            projections = RAGIndex._hash_term(term, vector_dim)
            df = doc_freq.get(term, 0)
            idf = safe_log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            weight = (1.0 + safe_log(float(tf) + 1.0)) * idf
            for idx, sign in projections:
                vector[idx] += weight * sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    @staticmethod
    def _hash_term(term: str, vector_dim: int) -> list[tuple[int, float]]:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=16).digest()
        idx1 = int.from_bytes(digest[:4], byteorder="big") % vector_dim
        idx2 = int.from_bytes(digest[4:8], byteorder="big") % vector_dim
        sign1 = 1.0 if digest[8] % 2 == 0 else -1.0
        sign2 = 1.0 if digest[9] % 2 == 0 else -1.0
        return [(idx1, sign1), (idx2, sign2)]


def format_search_hits(hits: list[SearchHit]) -> str:
    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        rerank = (
            f" | rerank={hit.rerank_score:.2f}"
            if hit.rerank_score is not None
            else ""
        )
        lines.append(
            f"{index}. [{hit.chunk.published}] {hit.chunk.title} | score={hit.score:.2f} | bm25={hit.bm25_score:.2f} | vec={hit.vector_score:.2f}{rerank}"
        )
        lines.append(f"   {snippet(hit.chunk.text, limit=150)}")
        lines.append(f"   {hit.chunk.url}")
    return "\n".join(lines)
