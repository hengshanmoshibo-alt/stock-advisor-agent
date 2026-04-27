from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


ScenarioKey = Literal["bullish", "neutral", "bearish"]


class ScenarioCard(BaseModel):
    key: ScenarioKey
    title: str
    stance: str
    reasoning: str
    risk: str


class Citation(BaseModel):
    title: str
    published: str
    url: str
    snippet: str
    score: float = Field(description="Hybrid retrieval score")


class SessionState(BaseModel):
    session_id: str
    turn_count: int
    summary: str
    model_mode: str


class TradePlanMetrics(BaseModel):
    ma20: float | None = None
    ma50: float | None = None
    ma200: float | None = None
    rsi14: float | None = None
    atr14: float | None = None
    high_60: float | None = None
    low_60: float | None = None
    high_120: float | None = None
    low_120: float | None = None
    high_252: float | None = None
    low_252: float | None = None
    weak_state: bool = False
    data_points: int = 0


class TradePlanVolume(BaseModel):
    average_volume_20: float | None = None
    average_volume_50: float | None = None
    volume_ratio_20: float | None = None
    volume_ratio_50: float | None = None
    signal: str = "unknown"
    note: str = ""


class TradePlanBacktest(BaseModel):
    sample_count: int = 0
    hit_rate_60d: float | None = None
    average_forward_return_60d: float | None = None
    average_max_drawdown_60d: float | None = None
    calibrated_first_buy_drawdown: list[float] = Field(default_factory=list)


class TradePlanScoreBreakdown(BaseModel):
    technical: int = 0
    volume: int = 0
    fundamentals: int = 0
    market: int = 0
    event_risk: int = 0
    total: int = 0
    reasons: list[str] = Field(default_factory=list)


class TradePlanCalibration(BaseModel):
    source_key: str = ""
    generated_at: str = ""
    parameter_source: str = "default_formula"
    confidence_adjustment: int = 0
    parameters: dict[str, Any] = Field(default_factory=dict)
    usable: bool = False


class TradePlanBacktestSummary(BaseModel):
    data_points: int = 0
    parameter_source: str = "default_formula"
    parameters: dict[str, Any] = Field(default_factory=dict)
    node_stats: dict[str, Any] = Field(default_factory=dict)


class TradePlanNode(BaseModel):
    key: str
    title: str
    active: bool
    lower: float | None = None
    upper: float | None = None
    action: str
    formula: str
    role_label: str | None = None
    plain_explanation: str | None = None


class TradePlanPayload(BaseModel):
    ticker: str
    display_stock: str
    current_price: float
    as_of: str
    action_state: str
    confidence: str
    risk_state: str
    note: str
    trend_stage: str | None = None
    metrics: TradePlanMetrics
    volume: TradePlanVolume | None = None
    backtest: TradePlanBacktest | None = None
    score_breakdown: TradePlanScoreBreakdown | None = None
    parameter_source: str = "default_formula"
    calibration: TradePlanCalibration | None = None
    backtest_summary: TradePlanBacktestSummary | None = None
    nodes: list[TradePlanNode]
    confirmation_condition: str
    failure_condition: str
    risk_adjustments: list[str]
    fundamentals: dict[str, Any] | None = None
    market_context: dict[str, Any] | None = None


class ChatPayload(BaseModel):
    answer: str
    scenarios: list[ScenarioCard]
    citations: list[Citation]
    disclaimer: str
    session_state: SessionState
    answer_mode: str = "generic_rag"
    trade_plan: TradePlanPayload | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=4000)


class SessionCreateResponse(BaseModel):
    session_id: str


class SessionListItem(BaseModel):
    session_id: str
    title: str
    updated_at: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    title: str
    messages: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    index_path: str
    articles: int
    chunks: int
    vector_dim: int
    vector_backend: str
    vector_model: str | None = None
    model_mode: str
    reranker_mode: str
