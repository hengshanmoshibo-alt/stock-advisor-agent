from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx
import numpy as np


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class DenseEmbedder:
    backend = "dense"
    model_name: str | None = None

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        matrix = self.embed_texts([text])
        return matrix[0]


class OpenAICompatibleEmbedder(DenseEmbedder):
    backend = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: float,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model_name, "input": list(texts)},
            )
            response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            raise ValueError("Embedding API returned no vectors")
        vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        return _normalize_matrix(vectors)


class TransformersEmbedder(DenseEmbedder):
    backend = "transformers"

    def __init__(self, model_name: str, *, device: str | None = None, batch_size: int = 16) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers embeddings require torch and transformers. "
                "Use EMBEDDING_BACKEND=hash for lightweight local runs."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        model_path = Path(model_name)
        local_files_only = model_path.exists()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        all_vectors: list[np.ndarray] = []
        text_list = list(texts)
        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            with self._torch.inference_mode():
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                masked = token_embeddings * attention_mask
                pooled = masked.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                vectors = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                all_vectors.append(vectors)
        return _normalize_matrix(np.vstack(all_vectors))


@dataclass(slots=True)
class EmbeddingSettings:
    backend: str
    model_name: str | None
    api_key: str | None
    base_url: str | None
    timeout: float
    hf_home: str | None = None


def build_dense_embedder(settings: EmbeddingSettings) -> DenseEmbedder | None:
    backend = settings.backend.lower().strip()
    if backend in {"", "hash", "none"}:
        return None
    if backend == "openai":
        if not (settings.api_key and settings.base_url and settings.model_name):
            raise ValueError("OpenAI-compatible embeddings require API key, base URL, and model")
        return OpenAICompatibleEmbedder(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model_name=settings.model_name,
            timeout=settings.timeout,
        )
    if backend == "transformers":
        model_name = settings.model_name or discover_local_embedding_path(settings.hf_home)
        if not model_name:
            raise ValueError("Transformers embeddings require EMBEDDING_MODEL or a local cached model")
        return TransformersEmbedder(model_name)
    raise ValueError(f"Unsupported embedding backend: {settings.backend}")


def discover_local_embedding_path(hf_home: str | None) -> str | None:
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
        "models--BAAI--bge-small-zh-v1.5",
        "models--BAAI--bge-base-zh-v1.5",
        "models--moka-ai--m3e-base",
        "models--thenlper--gte-base-zh",
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
