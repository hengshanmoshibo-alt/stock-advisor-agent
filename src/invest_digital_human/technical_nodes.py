from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .backtest_engine import BacktestReport, StrategyParameters, backtest_node_strategy
from .calibration import TradeNodeCalibration
from .fundamental_data import FundamentalSnapshot
from .market_data import MarketCandle
from .quote_lookup import QuoteSnapshot
from .trade_scoring import (
    BacktestCalibration,
    VolumeProfile,
    build_volume_profile,
    calibrate_from_history,
    classify_trend_stage,
)

if TYPE_CHECKING:
    from .market_context import MarketContext


CALCULATED_NODE_KEYWORDS = (
    "计算",
    "技术",
    "系统算",
    "自己算",
    "technical",
    "calculate",
    "entry",
)
CURRENT_NODE_KEYWORDS = ("现在", "当前", "实时", "最新")
NODE_TERMS = ("买点", "买入", "节点")


@dataclass(frozen=True, slots=True)
class TechnicalSnapshot:
    ticker: str
    as_of: str
    current_price: float
    ma20: float | None
    ma50: float | None
    ma200: float | None
    rsi14: float | None
    atr14: float | None
    high_60: float | None
    low_60: float | None
    high_120: float | None
    low_120: float | None
    high_252: float | None
    low_252: float | None
    weak_state: bool
    data_points: int

    def as_facts(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "current_price": round(self.current_price, 2),
            "ma20": _round_optional(self.ma20),
            "ma50": _round_optional(self.ma50),
            "ma200": _round_optional(self.ma200),
            "rsi14": _round_optional(self.rsi14),
            "atr14": _round_optional(self.atr14),
            "high_60": _round_optional(self.high_60),
            "low_60": _round_optional(self.low_60),
            "high_120": _round_optional(self.high_120),
            "low_120": _round_optional(self.low_120),
            "high_252": _round_optional(self.high_252),
            "low_252": _round_optional(self.low_252),
            "weak_state": self.weak_state,
            "data_points": self.data_points,
        }


@dataclass(frozen=True, slots=True)
class CalculatedNode:
    key: str
    title: str
    lower: float
    upper: float
    action: str
    formula: str

    def as_facts(self) -> dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "lower": round(self.lower, 2),
            "upper": round(self.upper, 2),
            "action": self.action,
            "formula": self.formula,
        }


@dataclass(frozen=True, slots=True)
class CalculatedNodePlan:
    ticker: str
    snapshot: TechnicalSnapshot
    nodes: list[CalculatedNode]
    risk_state: str
    note: str
    volume_profile: VolumeProfile
    trend_stage: str
    backtest_calibration: BacktestCalibration
    parameter_source: str
    calibration: TradeNodeCalibration | None
    backtest_summary: BacktestReport

    @property
    def data_sufficient(self) -> bool:
        return self.snapshot.data_points >= 200 and bool(self.nodes)

    def as_facts(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "risk_state": self.risk_state,
            "note": self.note,
            "snapshot": self.snapshot.as_facts(),
            "nodes": [node.as_facts() for node in self.nodes],
            "volume_profile": self.volume_profile.as_facts(),
            "trend_stage": self.trend_stage,
            "backtest_calibration": self.backtest_calibration.as_facts(),
            "parameter_source": self.parameter_source,
            "calibration": self.calibration.as_facts() if self.calibration else None,
            "backtest_summary": self.backtest_summary.as_facts(),
        }


def calculated_nodes_requested(query: str) -> bool:
    normalized = query.strip().lower()
    if any(keyword in normalized for keyword in CALCULATED_NODE_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in ("哪些", "有哪些", "历史")):
        return False
    if any(keyword in normalized for keyword in CURRENT_NODE_KEYWORDS) and any(
        term in normalized for term in NODE_TERMS
    ):
        return True
    return "是多少" in normalized and any(term in normalized for term in NODE_TERMS)


def calculate_technical_node_plan(
    *,
    candles: list[MarketCandle],
    quote: QuoteSnapshot,
    calibration: TradeNodeCalibration | None = None,
) -> CalculatedNodePlan | None:
    if not candles:
        return None
    ordered = sorted(candles, key=lambda item: item.timestamp)
    closes = [item.close for item in ordered]
    latest_price = quote.price
    snapshot = TechnicalSnapshot(
        ticker=quote.ticker,
        as_of=ordered[-1].date,
        current_price=latest_price,
        ma20=_moving_average(closes, 20),
        ma50=_moving_average(closes, 50),
        ma200=_moving_average(closes, 200),
        rsi14=_rsi(closes, 14),
        atr14=_atr(ordered, 14),
        high_60=_window_high(ordered, 60),
        low_60=_window_low(ordered, 60),
        high_120=_window_high(ordered, 120),
        low_120=_window_low(ordered, 120),
        high_252=_window_high(ordered, 252),
        low_252=_window_low(ordered, 252),
        weak_state=False,
        data_points=len(ordered),
    )
    weak_state = bool(
        (snapshot.ma200 is not None and latest_price < snapshot.ma200)
        or (snapshot.rsi14 is not None and snapshot.rsi14 < 35)
    )
    snapshot = TechnicalSnapshot(
        ticker=snapshot.ticker,
        as_of=snapshot.as_of,
        current_price=snapshot.current_price,
        ma20=snapshot.ma20,
        ma50=snapshot.ma50,
        ma200=snapshot.ma200,
        rsi14=snapshot.rsi14,
        atr14=snapshot.atr14,
        high_60=snapshot.high_60,
        low_60=snapshot.low_60,
        high_120=snapshot.high_120,
        low_120=snapshot.low_120,
        high_252=snapshot.high_252,
        low_252=snapshot.low_252,
        weak_state=weak_state,
        data_points=snapshot.data_points,
    )
    volume_profile = build_volume_profile(ordered)
    trend_stage = classify_trend_stage(snapshot)
    backtest_calibration = calibrate_from_history(ordered)
    parameter_source = "calibrated" if calibration is not None and calibration.usable() else "default_formula"
    strategy_parameters = calibration.parameters if parameter_source == "calibrated" else StrategyParameters(
        first_buy_drawdown_band=backtest_calibration.calibrated_first_buy_drawdown
    )
    backtest_summary = backtest_node_strategy(
        ordered,
        parameters=strategy_parameters,
        parameter_source=parameter_source,
    )
    nodes = _build_nodes(
        snapshot,
        trend_stage=trend_stage,
        calibration=backtest_calibration,
        parameters=strategy_parameters,
    )
    risk_state = "weak" if weak_state else "normal"
    note = (
        "Weak state: price is below MA200 or RSI14 is below 35; only observation/wait-for-confirmation nodes are active."
        if weak_state
        else "Normal state: calculated nodes use deterministic moving-average, drawdown, and ATR formulas."
    )
    return CalculatedNodePlan(
        ticker=quote.ticker,
        snapshot=snapshot,
        nodes=nodes,
        risk_state=risk_state,
        note=note,
        volume_profile=volume_profile,
        trend_stage=trend_stage,
        backtest_calibration=backtest_calibration,
        parameter_source=parameter_source,
        calibration=calibration if parameter_source == "calibrated" else None,
        backtest_summary=backtest_summary,
    )


def format_calculated_node_answer(plan: CalculatedNodePlan, *, historical_answer: str | None = None) -> str:
    lines = [
        f"{plan.ticker} technical node plan as of {plan.snapshot.as_of}: current price {plan.snapshot.current_price:.2f}.",
        _format_snapshot(plan.snapshot),
        "",
        "Calculated nodes:",
    ]
    if plan.nodes:
        for node in plan.nodes:
            lines.append(
                f"- {node.title}: {node.lower:.2f}-{node.upper:.2f}; action: {node.action}; formula: {node.formula}"
            )
    else:
        lines.append("- Not enough technical confirmation to produce active nodes.")
    if historical_answer:
        lines.extend(["", "Historical article nodes:", historical_answer])
    lines.extend(
        [
            "",
            f"Risk state: {plan.risk_state}. {plan.note}",
            "These are formula-based technical reference levels, not investment advice.",
        ]
    )
    return "\n".join(lines)


def format_enhanced_calculated_node_answer(
    plan: CalculatedNodePlan,
    *,
    historical_answer: str | None = None,
    fundamentals: FundamentalSnapshot | None = None,
    market_context: MarketContext | None = None,
) -> str:
    answer = format_calculated_node_answer(plan, historical_answer=historical_answer)
    sections: list[str] = []
    if fundamentals is not None:
        sections.append(_format_fundamental_context(fundamentals))
    if market_context is not None:
        sections.append(_format_market_context(market_context))
    if not sections:
        return answer
    return "\n\n".join([answer, "Additional confirmation data:", *sections])


def calculated_node_extra_facts(
    *,
    fundamentals: FundamentalSnapshot | None,
    market_context: MarketContext | None,
) -> dict[str, object]:
    return {
        "fundamentals": fundamentals.as_facts() if fundamentals else None,
        "market_context": market_context.as_facts() if market_context else None,
    }


def _format_snapshot(snapshot: TechnicalSnapshot) -> str:
    return (
        "Indicators: "
        f"MA20={_fmt(snapshot.ma20)}, MA50={_fmt(snapshot.ma50)}, MA200={_fmt(snapshot.ma200)}, "
        f"RSI14={_fmt(snapshot.rsi14)}, ATR14={_fmt(snapshot.atr14)}, "
        f"60D high/low={_fmt(snapshot.high_60)}/{_fmt(snapshot.low_60)}, "
        f"120D high/low={_fmt(snapshot.high_120)}/{_fmt(snapshot.low_120)}."
    )


def _format_fundamental_context(fundamentals: FundamentalSnapshot) -> str:
    lines = ["Fundamental/earnings context:"]
    metrics = fundamentals.metrics
    if metrics:
        selected = []
        for key in (
            "52WeekHigh",
            "52WeekLow",
            "52WeekPriceReturnDaily",
            "peBasicExclExtraTTM",
            "peNormalizedAnnual",
            "psTTM",
            "revenueGrowthTTMYoy",
            "epsGrowthTTMYoy",
        ):
            if key in metrics:
                selected.append(f"{key}={metrics[key]:.2f}")
        if selected:
            lines.append("- Metrics: " + ", ".join(selected))
    if fundamentals.latest_earnings is not None:
        item = fundamentals.latest_earnings
        lines.append(
            f"- Latest earnings: period={item.period}, actual={_fmt(item.actual)}, "
            f"estimate={_fmt(item.estimate)}, surprise={_fmt(item.surprise_percent)}%."
        )
    if fundamentals.next_earnings is not None:
        item = fundamentals.next_earnings
        days = fundamentals.days_to_earnings()
        timing = f", days_to_earnings={days}" if days is not None else ""
        lines.append(
            f"- Next earnings: date={item.date}, eps_estimate={_fmt(item.eps_estimate)}, "
            f"revenue_estimate={_fmt(item.revenue_estimate)}{timing}."
        )
        if days is not None and 0 <= days <= 7:
            lines.append("- Earnings risk: within 7 days; avoid treating technical zones as confirmed before the event.")
    if fundamentals.latest_recommendation is not None:
        item = fundamentals.latest_recommendation
        lines.append(
            f"- Analyst trend: strongBuy={item.strong_buy}, buy={item.buy}, hold={item.hold}, "
            f"sell={item.sell}, strongSell={item.strong_sell}."
        )
    return "\n".join(lines)


def _format_market_context(context: Any) -> str:
    lines = ["Market context:"]
    for item in context.indices:
        lines.append(
            f"- {item.ticker}: price={item.current_price:.2f}, MA50={_fmt(item.ma50)}, "
            f"MA200={_fmt(item.ma200)}, trend={item.trend}."
        )
    if context.risk_off:
        lines.append("- Market risk: at least one major ETF is below MA200; reduce confidence in aggressive entries.")
    return "\n".join(lines)


def _build_nodes(
    snapshot: TechnicalSnapshot,
    *,
    trend_stage: str,
    calibration: BacktestCalibration,
    parameters: StrategyParameters,
) -> list[CalculatedNode]:
    if snapshot.data_points < 200 or snapshot.ma50 is None or snapshot.atr14 is None:
        return []
    observation_lower = snapshot.ma50 - snapshot.atr14
    observation_upper = snapshot.ma50
    observation_formula = "MA50 到 MA50 - 1 * ATR14"
    observation_action = _observation_action(
        current_price=snapshot.current_price,
        lower=observation_lower,
        upper=observation_upper,
        strong_trend=False,
    )
    if trend_stage == "strong_uptrend" and snapshot.ma20 is not None:
        observation_lower = min(snapshot.ma20, snapshot.ma50)
        observation_upper = max(snapshot.ma20, snapshot.ma50)
        observation_formula = "Strong trend pullback zone: MA20 to MA50"
        observation_action = _observation_action(
            current_price=snapshot.current_price,
            lower=observation_lower,
            upper=observation_upper,
            strong_trend=True,
        )
    observation = CalculatedNode(
        key="observation",
        title="Observation zone",
        lower=min(observation_lower, observation_upper),
        upper=max(observation_lower, observation_upper),
        action=observation_action,
        formula=observation_formula,
    )
    if snapshot.weak_state:
        return [observation]
    nodes = [observation]
    if snapshot.high_60 is not None:
        lower_drawdown, upper_drawdown = parameters.first_buy_drawdown_band
        nodes.append(
            CalculatedNode(
                key="first_buy",
                title="第一买入区",
                lower=snapshot.high_60 * (1 - upper_drawdown),
                upper=snapshot.high_60 * (1 - lower_drawdown),
                action="只作为分批入场区域；需要回撤压力减缓后再使用。",
                formula=(
                    f"60 日高点回撤 {lower_drawdown:.0%}-{upper_drawdown:.0%}；"
                    "区间来自历史 60 日后续表现校准"
                ),
            )
        )
    if snapshot.high_120 is not None and snapshot.ma200 is not None:
        deep_lower_drawdown, deep_upper_drawdown = parameters.deep_buy_drawdown_band
        raw_lower = snapshot.high_120 * (1 - deep_upper_drawdown)
        raw_upper = snapshot.high_120 * (1 - deep_lower_drawdown)
        cap = snapshot.ma200 * 1.08
        upper = min(raw_upper, cap)
        lower = min(raw_lower, upper)
        nodes.append(
            CalculatedNode(
                key="deep_buy",
                title="Deep buy zone",
                lower=lower,
                upper=upper,
                action="仅用于更深度回撤；如果价格仍处于延伸状态，不追高。",
                formula=(
                    f"120 日高点回撤 {deep_lower_drawdown:.0%}-{deep_upper_drawdown:.0%}，"
                    "且不高于 MA200 * 1.08"
                ),
            )
        )
    if snapshot.ma200 is not None:
        defense = min(snapshot.ma200, snapshot.current_price - parameters.defense_atr_multiplier * snapshot.atr14)
        nodes.append(
            CalculatedNode(
                key="defense",
                title="Defense level",
                lower=defense,
                upper=defense,
                action="如果价格有效跌破该位置，降低风险或等待新的交易结构。",
                formula=f"min(MA200, 当前价 - {parameters.defense_atr_multiplier:g} * ATR14)",
            )
        )
    return nodes


def _observation_action(
    *,
    current_price: float,
    lower: float,
    upper: float,
    strong_trend: bool,
) -> str:
    low = min(lower, upper)
    high = max(lower, upper)
    if low <= current_price <= high:
        return "已进入观察区；先看能否企稳确认，不是立刻买入信号。"
    if current_price > high:
        if strong_trend:
            return "等待回踩到 MA20/MA50 附近并企稳；避免追高。"
        return "等待价格回到观察区并企稳；未进入前不追高。"
    return "价格已跌破观察区，先等待重新收回区间；不急于接下跌。"


def _moving_average(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _window_high(candles: list[MarketCandle], window: int) -> float | None:
    if len(candles) < window:
        return None
    return max(item.high for item in candles[-window:])


def _window_low(candles: list[MarketCandle], window: int) -> float | None:
    if len(candles) < window:
        return None
    return min(item.low for item in candles[-window:])


def _rsi(values: list[float], window: int) -> float | None:
    if len(values) <= window:
        return None
    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    recent = deltas[-window:]
    gains = [max(delta, 0.0) for delta in recent]
    losses = [abs(min(delta, 0.0)) for delta in recent]
    average_gain = sum(gains) / window
    average_loss = sum(losses) / window
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def _atr(candles: list[MarketCandle], window: int) -> float | None:
    if len(candles) <= window:
        return None
    true_ranges: list[float] = []
    for index in range(1, len(candles)):
        candle = candles[index]
        previous_close = candles[index - 1].close
        true_ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    recent = true_ranges[-window:]
    return sum(recent) / window


def _round_optional(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
