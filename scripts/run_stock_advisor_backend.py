from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("PYTHONPATH", str(SRC))
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("HUGGINGFACE_HOME", r"D:\home\software\huggingface")
os.environ.setdefault("RERANKER_STRATEGY", "none")
os.environ.setdefault("OLLAMA_MODEL", "gemma3:4b")
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
os.environ.setdefault("HOST", "127.0.0.1")
os.environ.setdefault("PORT", "8020")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


if __name__ == "__main__":
    uvicorn.run(
        "invest_digital_human.stock_advisor_api:app",
        host=os.environ["HOST"],
        port=int(os.environ["PORT"]),
        reload=False,
    )
