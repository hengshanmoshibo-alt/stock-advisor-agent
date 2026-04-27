from __future__ import annotations

from dataclasses import dataclass

from .market_data import MarketCandle


@dataclass(frozen=True, slots=True)
class StrategyParameters:
    first_buy_drawdown_band: tuple[float, float] = (0.15, 0.20)
    deep_buy_drawdown_band: tuple[float, float] = (0.25, 0.30)
    defense_atr_multiplier: float = 2.0
    min_sample_count: int = 8

    def as_facts(self) -> dict[str, object]:
        return {
            "first_buy_drawdown_band": [
                round(self.first_buy_drawdown_band[0], 4),
                round(self.first_buy_drawdown_band[1], 4),
            ],
            "deep_buy_drawdown_band": [
                round(self.deep_buy_drawdown_band[0], 4),
                round(self.deep_buy_drawdown_band[1], 4),
            ],
            "defense_atr_multiplier": round(self.defense_atr_multiplier, 4),
            "min_sample_count": self.min_sample_count,
        }


@dataclass(frozen=True, slots=True)
class NodeBacktestStats:
    node_key: str
    trigger_count: int
    hit_rate_20d: float | None
    hit_rate_60d: float | None
    hit_rate_120d: float | None
    average_return_20d: float | None
    average_return_60d: float | None
    average_return_120d: float | None
    average_max_drawdown_120d: float | None
    stop_loss_rate_120d: float | None

    def as_facts(self) -> dict[str, object]:
        return {
            "node_key": self.node_key,
            "trigger_count": self.trigger_count,
            "hit_rate_20d": _round_optional(self.hit_rate_20d),
            "hit_rate_60d": _round_optional(self.hit_rate_60d),
            "hit_rate_120d": _round_optional(self.hit_rate_120d),
            "average_return_20d": _round_optional(self.average_return_20d),
            "average_return_60d": _round_optional(self.average_return_60d),
            "average_return_120d": _round_optional(self.average_return_120d),
            "average_max_drawdown_120d": _round_optional(self.average_max_drawdown_120d),
            "stop_loss_rate_120d": _round_optional(self.stop_loss_rate_120d),
        }


@dataclass(frozen=True, slots=True)
class BacktestReport:
    data_points: int
    parameter_source: str
    parameters: StrategyParameters
    node_stats: dict[str, NodeBacktestStats]

    def as_facts(self) -> dict[str, object]:
        return {
            "data_points": self.data_points,
            "parameter_source": self.parameter_source,
            "parameters": self.parameters.as_facts(),
            "node_stats": {key: value.as_facts() for key, value in self.node_stats.items()},
        }


def backtest_node_strategy(
    candles: list[MarketCandle],
    *,
    parameters: StrategyParameters | None = None,
    parameter_source: str = "default_formula",
) -> BacktestReport:
    params = parameters or StrategyParameters()
    ordered = sorted(candles, key=lambda item: item.timestamp)
    samples: dict[str, list[_TradeOutcome]] = {"first_buy": [], "deep_buy": []}
    if len(ordered) < 321:
        return BacktestReport(
            data_points=len(ordered),
            parameter_source=parameter_source,
            parameters=params,
            node_stats={key: _empty_stats(key) for key in samples},
        )

    for index in range(200, len(ordered) - 120):
        history = ordered[: index + 1]
        candle = ordered[index]
        atr14 = _atr(history, 14)
        ma200 = _moving_average([item.close for item in history], 200)
        if atr14 is None or ma200 is None:
            continue
        high60 = max(item.high for item in ordered[index - 60 : index])
        high120 = max(item.high for item in ordered[index - 120 : index])
        defense = min(ma200, candle.close - params.defense_atr_multiplier * atr14)

        first = _zone_from_high(high60, params.first_buy_drawdown_band)
        if _touches_zone(candle, first):
            samples["first_buy"].append(_measure_outcome(ordered, index, _entry_price(first), defense))

        deep = _zone_from_high(high120, params.deep_buy_drawdown_band)
        if _touches_zone(candle, deep):
            samples["deep_buy"].append(_measure_outcome(ordered, index, _entry_price(deep), defense))

    return BacktestReport(
        data_points=len(ordered),
        parameter_source=parameter_source,
        parameters=params,
        node_stats={key: _summarize(key, values) for key, values in samples.items()},
    )


def select_first_buy_band_from_backtests(candles: list[MarketCandle]) -> tuple[float, float]:
    candidates = ((0.12, 0.18), (0.15, 0.20), (0.18, 0.25))
    scored: list[tuple[float, tuple[float, float]]] = []
    for band in candidates:
        report = backtest_node_strategy(
            candles,
            parameters=StrategyParameters(first_buy_drawdown_band=band),
            parameter_source="candidate",
        )
        stats = report.node_stats["first_buy"]
        if stats.trigger_count < 5 or stats.hit_rate_60d is None or stats.average_max_drawdown_120d is None:
            continue
        score = stats.hit_rate_60d + (stats.average_return_60d or 0) + max(stats.average_max_drawdown_120d, -0.5)
        scored.append((score, band))
    if not scored:
        return StrategyParameters().first_buy_drawdown_band
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


@dataclass(frozen=True, slots=True)
class _TradeOutcome:
    return_20d: float
    return_60d: float
    return_120d: float
    max_drawdown_120d: float
    stopped: bool


def _measure_outcome(
    candles: list[MarketCandle],
    index: int,
    entry_price: float,
    defense: float,
) -> _TradeOutcome:
    window = candles[index : index + 121]
    return _TradeOutcome(
        return_20d=(candles[index + 20].close - entry_price) / entry_price,
        return_60d=(candles[index + 60].close - entry_price) / entry_price,
        return_120d=(candles[index + 120].close - entry_price) / entry_price,
        max_drawdown_120d=(min(item.low for item in window) - entry_price) / entry_price,
        stopped=any(item.low <= defense for item in window),
    )


def _summarize(node_key: str, outcomes: list[_TradeOutcome]) -> NodeBacktestStats:
    if not outcomes:
        return _empty_stats(node_key)
    return NodeBacktestStats(
        node_key=node_key,
        trigger_count=len(outcomes),
        hit_rate_20d=_hit_rate([item.return_20d for item in outcomes]),
        hit_rate_60d=_hit_rate([item.return_60d for item in outcomes]),
        hit_rate_120d=_hit_rate([item.return_120d for item in outcomes]),
        average_return_20d=_average([item.return_20d for item in outcomes]),
        average_return_60d=_average([item.return_60d for item in outcomes]),
        average_return_120d=_average([item.return_120d for item in outcomes]),
        average_max_drawdown_120d=_average([item.max_drawdown_120d for item in outcomes]),
        stop_loss_rate_120d=sum(1 for item in outcomes if item.stopped) / len(outcomes),
    )


def _empty_stats(node_key: str) -> NodeBacktestStats:
    return NodeBacktestStats(node_key, 0, None, None, None, None, None, None, None, None)


def _zone_from_high(high: float, band: tuple[float, float]) -> tuple[float, float]:
    lower_drawdown, upper_drawdown = sorted(band)
    return (high * (1 - upper_drawdown), high * (1 - lower_drawdown))


def _touches_zone(candle: MarketCandle, zone: tuple[float, float]) -> bool:
    lower, upper = zone
    return candle.low <= upper and candle.high >= lower


def _entry_price(zone: tuple[float, float]) -> float:
    return (zone[0] + zone[1]) / 2


def _moving_average(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


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
    return sum(true_ranges[-window:]) / window


def _hit_rate(values: list[float]) -> float:
    return sum(1 for value in values if value > 0) / len(values)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
