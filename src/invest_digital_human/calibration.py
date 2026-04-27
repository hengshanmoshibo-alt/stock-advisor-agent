from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backtest_engine import (
    BacktestReport,
    NodeBacktestStats,
    StrategyParameters,
    backtest_node_strategy,
    select_first_buy_band_from_backtests,
)
from .market_data import MarketCandle


DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "data" / "calibration" / "trade_node_calibration.json"


@dataclass(frozen=True, slots=True)
class TradeNodeCalibration:
    source_key: str
    generated_at: str
    parameters: StrategyParameters
    backtest_report: BacktestReport
    confidence_adjustment: int = 0
    parameter_source: str = "calibrated"

    def usable(self) -> bool:
        first_buy = self.backtest_report.node_stats.get("first_buy")
        return bool(first_buy and first_buy.trigger_count >= self.parameters.min_sample_count)

    def as_facts(self) -> dict[str, object]:
        return {
            "source_key": self.source_key,
            "generated_at": self.generated_at,
            "parameter_source": self.parameter_source,
            "confidence_adjustment": self.confidence_adjustment,
            "parameters": self.parameters.as_facts(),
            "usable": self.usable(),
        }


class TradeNodeCalibrationStore:
    def __init__(self, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
        self.path = path
        self._items = load_calibrations(path)

    def get(self, source_key: str) -> TradeNodeCalibration | None:
        item = self._items.get(source_key.strip().lower())
        if item is None or not item.usable():
            return None
        return item


def build_calibration(source_key: str, candles: list[MarketCandle]) -> TradeNodeCalibration | None:
    ordered = sorted(candles, key=lambda item: item.timestamp)
    if len(ordered) < 321:
        return None
    first_buy_band = select_first_buy_band_from_backtests(ordered)
    params = StrategyParameters(first_buy_drawdown_band=first_buy_band)
    report = backtest_node_strategy(ordered, parameters=params, parameter_source="calibrated")
    first_buy = report.node_stats.get("first_buy")
    if first_buy is None or first_buy.trigger_count < params.min_sample_count:
        return None
    confidence_adjustment = 0
    if first_buy.hit_rate_60d is not None and first_buy.hit_rate_60d >= 0.58:
        confidence_adjustment = 1
    elif first_buy.hit_rate_60d is not None and first_buy.hit_rate_60d < 0.45:
        confidence_adjustment = -1
    return TradeNodeCalibration(
        source_key=source_key.strip().lower(),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        parameters=params,
        backtest_report=report,
        confidence_adjustment=confidence_adjustment,
    )


def load_calibrations(path: Path) -> dict[str, TradeNodeCalibration]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, dict):
        return {}
    result: dict[str, TradeNodeCalibration] = {}
    for key, value in items.items():
        if not isinstance(value, dict):
            continue
        item = _parse_calibration(str(key), value)
        if item is not None:
            result[item.source_key] = item
    return result


def save_calibrations(path: Path, calibrations: dict[str, TradeNodeCalibration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "items": {key: _serialize_calibration(value) for key, value in sorted(calibrations.items())},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _serialize_calibration(item: TradeNodeCalibration) -> dict[str, object]:
    return {
        "source_key": item.source_key,
        "generated_at": item.generated_at,
        "parameter_source": item.parameter_source,
        "confidence_adjustment": item.confidence_adjustment,
        "parameters": item.parameters.as_facts(),
        "backtest_report": item.backtest_report.as_facts(),
    }


def _parse_calibration(key: str, value: dict[str, Any]) -> TradeNodeCalibration | None:
    try:
        params = _parse_parameters(value.get("parameters"))
        report = _parse_report(value.get("backtest_report"), params)
        return TradeNodeCalibration(
            source_key=str(value.get("source_key") or key).strip().lower(),
            generated_at=str(value.get("generated_at") or ""),
            parameters=params,
            backtest_report=report,
            confidence_adjustment=int(value.get("confidence_adjustment") or 0),
            parameter_source=str(value.get("parameter_source") or "calibrated"),
        )
    except (TypeError, ValueError, KeyError):
        return None


def _parse_parameters(value: object) -> StrategyParameters:
    data = value if isinstance(value, dict) else {}
    return StrategyParameters(
        first_buy_drawdown_band=_pair(data.get("first_buy_drawdown_band"), (0.15, 0.20)),
        deep_buy_drawdown_band=_pair(data.get("deep_buy_drawdown_band"), (0.25, 0.30)),
        defense_atr_multiplier=float(data.get("defense_atr_multiplier") or 2.0),
        min_sample_count=int(data.get("min_sample_count") or 8),
    )


def _parse_report(value: object, parameters: StrategyParameters) -> BacktestReport:
    data = value if isinstance(value, dict) else {}
    raw_stats = data.get("node_stats")
    node_stats: dict[str, NodeBacktestStats] = {}
    if isinstance(raw_stats, dict):
        for key, item in raw_stats.items():
            if isinstance(item, dict):
                node_stats[str(key)] = _parse_node_stats(str(key), item)
    return BacktestReport(
        data_points=int(data.get("data_points") or 0),
        parameter_source=str(data.get("parameter_source") or "calibrated"),
        parameters=parameters,
        node_stats=node_stats,
    )


def _parse_node_stats(key: str, value: dict[str, Any]) -> NodeBacktestStats:
    return NodeBacktestStats(
        node_key=str(value.get("node_key") or key),
        trigger_count=int(value.get("trigger_count") or 0),
        hit_rate_20d=_float_or_none(value.get("hit_rate_20d")),
        hit_rate_60d=_float_or_none(value.get("hit_rate_60d")),
        hit_rate_120d=_float_or_none(value.get("hit_rate_120d")),
        average_return_20d=_float_or_none(value.get("average_return_20d")),
        average_return_60d=_float_or_none(value.get("average_return_60d")),
        average_return_120d=_float_or_none(value.get("average_return_120d")),
        average_max_drawdown_120d=_float_or_none(value.get("average_max_drawdown_120d")),
        stop_loss_rate_120d=_float_or_none(value.get("stop_loss_rate_120d")),
    )


def _pair(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return default
    return (float(value[0]), float(value[1]))


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
