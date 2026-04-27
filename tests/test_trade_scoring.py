from __future__ import annotations

import unittest
from types import SimpleNamespace

from invest_digital_human.market_data import MarketCandle
from invest_digital_human.trade_scoring import (
    VolumeProfile,
    build_volume_profile,
    calibrate_from_history,
    score_trade_plan,
    sector_etfs_for_source,
)


class TradeScoringTest(unittest.TestCase):
    def test_volume_profile_detects_breakout_and_distribution(self) -> None:
        breakout = build_volume_profile(_volume_candles(latest_close=121, previous_close=120))
        distribution = build_volume_profile(_volume_candles(latest_close=119, previous_close=120))

        self.assertEqual(breakout.signal, "breakout_volume")
        self.assertGreater(breakout.volume_ratio_20 or 0, 1.5)
        self.assertEqual(distribution.signal, "distribution_risk")

    def test_backtest_calibration_returns_stable_band(self) -> None:
        calibration = calibrate_from_history(_drawdown_candles(220))

        self.assertGreater(calibration.sample_count, 0)
        self.assertEqual(len(calibration.calibrated_first_buy_drawdown), 2)
        self.assertGreaterEqual(calibration.calibrated_first_buy_drawdown[0], 0.12)
        self.assertLessEqual(calibration.calibrated_first_buy_drawdown[1], 0.25)

    def test_score_combines_technical_volume_and_market(self) -> None:
        score = score_trade_plan(
            trend_stage="strong_uptrend",
            weak_state=False,
            rsi14=55,
            volume=VolumeProfile(
                average_volume_20=100,
                average_volume_50=100,
                volume_ratio_20=0.7,
                volume_ratio_50=0.7,
                signal="quiet_pullback",
                note="quiet pullback",
            ),
            fundamentals=None,
            market_context=SimpleNamespace(risk_off=False),
        )

        self.assertGreaterEqual(score.technical, 2)
        self.assertEqual(score.volume, 1)
        self.assertEqual(score.market, 1)
        self.assertGreater(score.total, 0)

    def test_sector_etf_mapping_prefers_related_etfs(self) -> None:
        self.assertEqual(sector_etfs_for_source("NVDA"), ["QQQ", "SMH"])
        self.assertEqual(sector_etfs_for_source(" msft "), ["QQQ", "XLK"])
        self.assertEqual(sector_etfs_for_source("unknown"), ["SPY", "QQQ"])


def _volume_candles(*, latest_close: float, previous_close: float) -> list[MarketCandle]:
    candles = [
        MarketCandle(
            timestamp=1_700_000_000 + index * 86_400,
            date=f"2025-01-{(index % 28) + 1:02d}",
            open=100,
            high=101,
            low=99,
            close=100 + index * 0.1,
            volume=100,
        )
        for index in range(49)
    ]
    candles[-1] = _candle(48, previous_close, 100)
    candles.append(_candle(49, latest_close, 300))
    return candles


def _drawdown_candles(count: int) -> list[MarketCandle]:
    candles: list[MarketCandle] = []
    for index in range(count):
        cycle = index % 40
        close = 115 - cycle if 10 <= cycle <= 25 else 105 + cycle * 0.5
        candles.append(
            MarketCandle(
                timestamp=1_700_000_000 + index * 86_400,
                date=f"2025-01-{(index % 28) + 1:02d}",
                open=close,
                high=close + 2,
                low=close - 2,
                close=close,
                volume=1_000_000,
            )
        )
    return candles


def _candle(index: int, close: float, volume: float) -> MarketCandle:
    return MarketCandle(
        timestamp=1_700_000_000 + index * 86_400,
        date=f"2025-01-{(index % 28) + 1:02d}",
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=volume,
    )


if __name__ == "__main__":
    unittest.main()
