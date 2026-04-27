from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .retrieval import SearchHit


class BaseReranker:
    mode = "none"

    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        return hits[:limit]


class LocalTransformersReranker(BaseReranker):
    mode = "local-transformers"

    def __init__(self, model_path: str, *, batch_size: int = 8, device: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Local reranker requires torch and transformers. "
                "Use RERANKER_STRATEGY=none for lightweight local runs."
            ) from exc

        self._torch = torch
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def rerank(self, query: str, hits: list[SearchHit], *, limit: int) -> list[SearchHit]:
        if not hits:
            return []
        tokenizer, model = self._ensure_loaded()
        reranked: list[SearchHit] = []
        for start in range(0, len(hits), self.batch_size):
            batch = hits[start : start + self.batch_size]
            pairs = [(query, hit.chunk.text) for hit in batch]
            with self._torch.inference_mode():
                encoded = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = model(**encoded).logits.view(-1).detach().cpu().tolist()
            for hit, score in zip(batch, logits):
                reranked.append(
                    SearchHit(
                        score=hit.score,
                        chunk=hit.chunk,
                        bm25_score=hit.bm25_score,
                        vector_score=hit.vector_score,
                        rerank_score=float(score),
                    )
                )

        reranked.sort(
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else float("-inf"),
                item.score,
            ),
            reverse=True,
        )
        return reranked[:limit]

    def _ensure_loaded(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._model.eval()
            self._model.to(self.device)
        return self._tokenizer, self._model


@dataclass(slots=True)
class RerankerSettings:
    strategy: str
    model_path: str | None
    hf_home: str | None


def build_reranker(settings: RerankerSettings) -> BaseReranker:
    strategy = settings.strategy.lower().strip()
    if strategy in {"", "none", "off"}:
        return BaseReranker()
    if strategy in {"local", "transformers"}:
        model_path = settings.model_path or discover_local_reranker_path(settings.hf_home)
        if not model_path:
            raise ValueError("Could not locate a local reranker model")
        return LocalTransformersReranker(model_path)
    raise ValueError(f"Unsupported reranker strategy: {settings.strategy}")


def discover_local_reranker_path(hf_home: str | None) -> str | None:
    roots = []
    if hf_home:
        roots.append(Path(hf_home))
    roots.extend(
        [
            Path(r"D:\home\software\huggingface"),
            Path.home() / ".cache" / "huggingface",
        ]
    )
    candidates = [
        "models--BAAI--bge-reranker-base",
        "models--BAAI--bge-reranker-large",
    ]
    for root in roots:
        hub = root / "hub"
        if not hub.exists():
            continue
        for model_dir_name in candidates:
            snapshot_root = hub / model_dir_name / "snapshots"
            if not snapshot_root.exists():
                continue
            snapshots = sorted(
                [path for path in snapshot_root.iterdir() if path.is_dir()],
                key=lambda item: item.name,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0])
    return None
