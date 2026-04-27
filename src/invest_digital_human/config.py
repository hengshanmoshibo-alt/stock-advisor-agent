from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index.json"
DEFAULT_STOCK_NODE_PATH = PROJECT_ROOT / "data" / "final_finetune" / "buy_nodes_master.jsonl"
DEFAULT_TRADE_NODE_CALIBRATION_PATH = PROJECT_ROOT / "data" / "calibration" / "trade_node_calibration.json"
LOCAL_DEFAULTS = {
    "QUOTE_PROVIDER": "finnhub",
    "MARKET_DATA_PROVIDER": "massive",
}


def load_local_env(path: Path | None = None) -> None:
    """Load a local .env file without overriding explicit environment values."""

    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        key = name.strip()
        if not key or key in os.environ:
            continue
        parsed = value.strip().strip('"').strip("'")
        os.environ[key] = parsed


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    for name in names:
        value = LOCAL_DEFAULTS.get(name)
        if value:
            return value
    return None


@dataclass(slots=True)
class AppSettings:
    index_path: Path = DEFAULT_INDEX_PATH
    stock_node_path: Path = DEFAULT_STOCK_NODE_PATH
    trade_node_calibration_path: Path = DEFAULT_TRADE_NODE_CALIBRATION_PATH
    enable_stock_node_rag: bool = True
    stock_node_max_items: int = 8
    enable_quote_lookup: bool = True
    enable_trade_plan_llm: bool = True
    quote_lookup_timeout: float = 8.0
    quote_provider: str = "finnhub"
    finnhub_api_key: str | None = None
    market_data_provider: str = "massive"
    massive_api_key: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None
    top_k: int = 5
    recall_limit: int = 12
    search_mode: str = "hybrid"
    session_window: int = 6
    summary_threshold: int = 10
    request_timeout: float = 60.0
    host: str = "127.0.0.1"
    port: int = 8000
    embedding_backend: str = "hash"
    embedding_model: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    reranker_strategy: str = "none"
    reranker_model_path: str | None = None
    huggingface_home: str | None = None

    @classmethod
    def from_env(cls) -> "AppSettings":
        load_local_env()
        index_path = Path(
            _first_env("INVEST_DH_INDEX_PATH", "INDEX_PATH") or str(DEFAULT_INDEX_PATH)
        )
        api_key = _first_env("INVEST_DH_API_KEY", "API_KEY")
        base_url = _first_env("INVEST_DH_BASE_URL", "BASE_URL")
        return cls(
            index_path=index_path,
            stock_node_path=Path(
                _first_env("INVEST_DH_STOCK_NODE_PATH", "STOCK_NODE_PATH")
                or str(DEFAULT_STOCK_NODE_PATH)
            ),
            trade_node_calibration_path=Path(
                _first_env("INVEST_DH_TRADE_NODE_CALIBRATION_PATH", "TRADE_NODE_CALIBRATION_PATH")
                or str(DEFAULT_TRADE_NODE_CALIBRATION_PATH)
            ),
            enable_stock_node_rag=(
                (_first_env("INVEST_DH_ENABLE_STOCK_NODE_RAG", "ENABLE_STOCK_NODE_RAG") or "true").lower()
                not in {"0", "false", "off", "no"}
            ),
            stock_node_max_items=int(
                _first_env("INVEST_DH_STOCK_NODE_MAX_ITEMS", "STOCK_NODE_MAX_ITEMS") or 8
            ),
            enable_quote_lookup=(
                (_first_env("INVEST_DH_ENABLE_QUOTE_LOOKUP", "ENABLE_QUOTE_LOOKUP") or "true").lower()
                not in {"0", "false", "off", "no"}
            ),
            enable_trade_plan_llm=(
                (_first_env("INVEST_DH_ENABLE_TRADE_PLAN_LLM", "ENABLE_TRADE_PLAN_LLM") or "true").lower()
                not in {"0", "false", "off", "no"}
            ),
            quote_lookup_timeout=float(
                _first_env("INVEST_DH_QUOTE_LOOKUP_TIMEOUT", "QUOTE_LOOKUP_TIMEOUT") or 8.0
            ),
            quote_provider=(
                _first_env("INVEST_DH_QUOTE_PROVIDER", "QUOTE_PROVIDER") or "yahoo"
            ),
            finnhub_api_key=_first_env("INVEST_DH_FINNHUB_API_KEY", "FINNHUB_API_KEY"),
            market_data_provider=(
                _first_env("INVEST_DH_MARKET_DATA_PROVIDER", "MARKET_DATA_PROVIDER")
                or "finnhub"
            ),
            massive_api_key=_first_env("INVEST_DH_MASSIVE_API_KEY", "MASSIVE_API_KEY"),
            api_key=api_key,
            base_url=base_url,
            model=_first_env("INVEST_DH_MODEL", "MODEL"),
            ollama_base_url=_first_env("INVEST_DH_OLLAMA_BASE_URL", "OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434",
            ollama_model=_first_env("INVEST_DH_OLLAMA_MODEL", "OLLAMA_MODEL")
            or "gemma3:4b",
            top_k=int(_first_env("INVEST_DH_TOP_K", "TOP_K") or 5),
            recall_limit=int(_first_env("INVEST_DH_RECALL_LIMIT", "RECALL_LIMIT") or 12),
            search_mode=_first_env("INVEST_DH_SEARCH_MODE", "SEARCH_MODE") or "hybrid",
            session_window=int(_first_env("INVEST_DH_SESSION_WINDOW", "SESSION_WINDOW") or 6),
            summary_threshold=int(
                _first_env("INVEST_DH_SUMMARY_THRESHOLD", "SUMMARY_THRESHOLD") or 10
            ),
            request_timeout=float(
                _first_env("INVEST_DH_REQUEST_TIMEOUT", "REQUEST_TIMEOUT") or 60.0
            ),
            host=_first_env("INVEST_DH_HOST", "HOST") or "127.0.0.1",
            port=int(_first_env("INVEST_DH_PORT", "PORT") or 8000),
            embedding_backend=_first_env(
                "INVEST_DH_EMBEDDING_BACKEND",
                "EMBEDDING_BACKEND",
            )
            or "hash",
            embedding_model=_first_env(
                "INVEST_DH_EMBEDDING_MODEL",
                "EMBEDDING_MODEL",
            ),
            embedding_api_key=_first_env(
                "INVEST_DH_EMBEDDING_API_KEY",
                "EMBEDDING_API_KEY",
            )
            or api_key,
            embedding_base_url=_first_env(
                "INVEST_DH_EMBEDDING_BASE_URL",
                "EMBEDDING_BASE_URL",
            )
            or base_url,
            reranker_strategy=_first_env(
                "INVEST_DH_RERANKER_STRATEGY",
                "RERANKER_STRATEGY",
            )
            or "none",
            reranker_model_path=_first_env(
                "INVEST_DH_RERANKER_MODEL_PATH",
                "RERANKER_MODEL_PATH",
            ),
            huggingface_home=_first_env(
                "INVEST_DH_HUGGINGFACE_HOME",
                "HUGGINGFACE_HOME",
            )
            or r"D:\home\software\huggingface",
        )


def discover_default_article_dir() -> Path:
    configured = _first_env("INVEST_DH_ARTICLE_DIR", "ARTICLE_DIR")
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return candidate

    search_roots = [
        PROJECT_ROOT,
        PROJECT_ROOT.parent,
        PROJECT_ROOT.parent / "wechat-article-mcp",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        manifest = next(root.rglob("_manifest.json"), None)
        if manifest is not None:
            return manifest.parent

    raise FileNotFoundError(
        "Could not find the default article directory. "
        "Set ARTICLE_DIR or pass --articles explicitly."
    )
