from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .fundamental_data import FundamentalSnapshot
from .market_data import MarketCandle

if TYPE_CHECKING:
    from .market_context import MarketContext


SECTOR_ETF_BY_SOURCE = {
    "aapl": ("QQQ", "XLK"),
    "msft": ("QQQ", "XLK"),
    "nvda": ("QQQ", "SMH"),
    "amd": ("QQQ", "SMH"),
    "tsm": ("QQQ", "SMH"),
    "amzn": ("QQQ", "XLY"),
    "tsla": ("QQQ", "XLY"),
    "googl": ("QQQ", "XLC"),
    "meta": ("QQQ", "XLC"),
    "wmt": ("SPY", "XLP"),
    "costco": ("SPY", "XLP"),
    "ko": ("SPY", "XLP"),
}


@dataclass(frozen=True, slots=True)
class VolumeProfile:
    average_volume_20: float | None
    average_volume_50: float | None
    volume_ratio_20: float | None
    volume_ratio_50: float | None
    signal: str
    note: str

    def as_facts(self) -> dict[str, object]:
        return {
            "average_volume_20": _round_optional(self.average_volume_20),
            "average_volume_50": _round_optional(self.average_volume_50),
            "volume_ratio_20": _round_optional(self.volume_ratio_20),
            "volume_ratio_50": _round_optional(self.volume_ratio_50),
            "signal": self.signal,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class BacktestCalibration:
    sample_count: int
    hit_rate_60d: float | None
    average_forward_return_60d: float | None
    average_max_drawdown_60d: float | None
    calibrated_first_buy_drawdown: tuple[float, float]

    def as_facts(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "hit_rate_60d": _round_optional(self.hit_rate_60d),
            "average_forward_return_60d": _round_optional(self.average_forward_return_60d),
            "average_max_drawdown_60d": _round_optional(self.average_max_drawdown_60d),
            "calibrated_first_buy_drawdown": [
                round(self.calibrated_first_buy_drawdown[0], 4),
                round(self.calibrated_first_buy_drawdown[1], 4),
            ],
        }


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    technical: int
    volume: int
    fundamentals: int
    market: int
    event_risk: int
    total: int
    reasons: list[str]

    def as_facts(self) -> dict[str, object]:
        return {
            "technical": self.technical,
            "volume": self.volume,
            "fundamentals": self.fundamentals,
            "market": self.market,
            "event_risk": self.event_risk,
            "total": self.total,
            "reasons": self.reasons,
        }


def build_volume_profile(candles: list[MarketCandle]) -> VolumeProfile:
    ordered = sorted(candles, key=lambda item: item.timestamp)
    if len(ordered) < 20:
        return VolumeProfile(None, None, None, None, "insufficient", "成交量数据不足。")
    latest = ordered[-1]
    avg20 = _average([item.volume for item in ordered[-20:]])
    avg50 = _average([item.volume for item in ordered[-50:]]) if len(ordered) >= 50 else None
    ratio20 = latest.volume / avg20 if avg20 else None
    ratio50 = latest.volume / avg50 if avg50 else None
    previous_close = ordered[-2].close if len(ordered) >= 2 else latest.close

    if ratio20 is not None and ratio20 >= 1.5 and latest.close > previous_close:
        signal = "breakout_volume"
        note = "上涨日成交量显著高于20日均量，偏向放量确认。"
    elif ratio20 is not None and ratio20 >= 1.5 and latest.close < previous_close:
        signal = "distribution_risk"
        note = "下跌日成交量显著放大，存在派发或恐慌风险。"
    elif ratio20 is not None and ratio20 <= 0.8:
        signal = "quiet_pullback"
        note = "成交量低于20日均量，若价格回撤，偏向缩量回踩观察。"
    else:
        signal = "neutral"
        note = "成交量没有明显确认或风险信号。"

    return VolumeProfile(avg20, avg50, ratio20, ratio50, signal, note)


def classify_trend_stage(snapshot) -> str:
    price = snapshot.current_price
    ma20 = snapshot.ma20
    ma50 = snapshot.ma50
    ma200 = snapshot.ma200
    rsi14 = snapshot.rsi14
    if ma200 is not None and price < ma200:
        return "weak_below_ma200"
    if rsi14 is not None and rsi14 >= 75:
        return "extended_overheated"
    if ma20 is not None and ma50 is not None and ma200 is not None:
        if price > ma20 > ma50 > ma200:
            return "strong_uptrend"
        if price > ma50 > ma200:
            return "normal_uptrend"
        if ma200 < price < ma50:
            return "pullback_above_ma200"
    return "range_or_unclear"


def calibrate_from_history(candles: list[MarketCandle]) -> BacktestCalibration:
    ordered = sorted(candles, key=lambda item: item.timestamp)
    if len(ordered) < 140:
        return BacktestCalibration(0, None, None, None, (0.15, 0.20))
    samples: list[tuple[float, float]] = []
    for index in range(60, len(ordered) - 60):
        previous_high = max(item.high for item in ordered[index - 60 : index])
        if previous_high <= 0:
            continue
        close = ordered[index].close
        drawdown = (previous_high - close) / previous_high
        if 0.10 <= drawdown <= 0.35:
            future = ordered[index + 60]
            window = ordered[index : index + 61]
            forward_return = (future.close - close) / close
            max_drawdown = (min(item.low for item in window) - close) / close
            samples.append((forward_return, max_drawdown))
    if not samples:
        return BacktestCalibration(0, None, None, None, (0.15, 0.20))

    avg_return = _average([item[0] for item in samples])
    avg_drawdown = _average([item[1] for item in samples])
    hit_rate = sum(1 for item in samples if item[0] > 0) / len(samples)
    if hit_rate >= 0.55 and avg_drawdown > -0.12:
        band = (0.12, 0.18)
    elif hit_rate < 0.45 or avg_drawdown < -0.18:
        band = (0.18, 0.25)
    else:
        band = (0.15, 0.20)
    return BacktestCalibration(len(samples), hit_rate, avg_return, avg_drawdown, band)


def score_trade_plan(
    *,
    trend_stage: str,
    weak_state: bool,
    rsi14: float | None,
    volume: VolumeProfile,
    fundamentals: FundamentalSnapshot | None,
    market_context: "MarketContext | None",
) -> ScoreBreakdown:
    technical = 0
    volume_score = 0
    fundamental_score = 0
    market_score = 0
    event_risk = 0
    reasons: list[str] = []

    if trend_stage in {"strong_uptrend", "normal_uptrend"}:
        technical += 2
        reasons.append(f"趋势阶段为 {trend_stage}，技术结构加分。")
    elif trend_stage == "pullback_above_ma200":
        technical += 1
        reasons.append("价格在MA200上方回撤，技术结构小幅加分。")
    elif trend_stage == "weak_below_ma200" or weak_state:
        technical -= 2
        reasons.append("价格跌破MA200或处于弱势，技术结构扣分。")
    if rsi14 is not None and rsi14 >= 75:
        technical -= 1
        reasons.append("RSI14过热，避免追高。")

    if volume.signal == "breakout_volume":
        volume_score += 1
        reasons.append("成交量放大且上涨，量能确认加分。")
    elif volume.signal == "quiet_pullback":
        volume_score += 1
        reasons.append("缩量回撤特征，适合等待止跌确认。")
    elif volume.signal == "distribution_risk":
        volume_score -= 2
        reasons.append("下跌放量，量能风险扣分。")

    if fundamentals is not None:
        if fundamentals.latest_earnings and fundamentals.latest_earnings.surprise_percent is not None:
            if fundamentals.latest_earnings.surprise_percent > 0:
                fundamental_score += 1
                reasons.append("最近EPS surprise为正，基本面加分。")
            elif fundamentals.latest_earnings.surprise_percent < 0:
                fundamental_score -= 1
                reasons.append("最近EPS surprise为负，基本面扣分。")
        if fundamentals.latest_recommendation:
            item = fundamentals.latest_recommendation
            if item.positive_count >= item.hold + item.negative_count:
                fundamental_score += 1
                reasons.append("分析师正向评级占优。")
            if item.negative_count > 0:
                fundamental_score -= 1
                reasons.append("分析师评级中存在卖出倾向。")
        days = fundamentals.days_to_earnings()
        if days is not None and 0 <= days <= 7:
            event_risk -= 2
            reasons.append(f"距离财报只有 {days} 天，事件风险扣分。")

    if market_context is not None:
        if market_context.risk_off:
            market_score -= 2
            reasons.append("SPY或QQQ跌破MA200，市场环境扣分。")
        else:
            market_score += 1
            reasons.append("SPY/QQQ未触发MA200风险，市场环境小幅加分。")

    total = technical + volume_score + fundamental_score + market_score + event_risk
    return ScoreBreakdown(technical, volume_score, fundamental_score, market_score, event_risk, total, reasons)


def sector_etfs_for_source(source_key: str) -> list[str]:
    return list(SECTOR_ETF_BY_SOURCE.get(source_key.strip().lower(), ("SPY", "QQQ")))


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
