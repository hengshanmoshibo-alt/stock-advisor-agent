from __future__ import annotations

import unittest

from invest_digital_human.backtest_engine import StrategyParameters, backtest_node_strategy
from invest_digital_human.market_data import MarketCandle


class BacktestEngineTest(unittest.TestCase):
    def test_backtest_outputs_stable_node_stats(self) -> None:
        report = backtest_node_strategy(_cyclical_candles(380), parameters=StrategyParameters())

        first_buy = report.node_stats["first_buy"]
        self.assertEqual(report.parameter_source, "default_formula")
        self.assertGreater(first_buy.trigger_count, 0)
        self.assertIsNotNone(first_buy.hit_rate_20d)
        self.assertIsNotNone(first_buy.average_return_60d)
        self.assertIsNotNone(first_buy.average_max_drawdown_120d)

    def test_insufficient_data_returns_empty_stats(self) -> None:
        report = backtest_node_strategy(_cyclical_candles(120))

        self.assertEqual(report.data_points, 120)
        self.assertEqual(report.node_stats["first_buy"].trigger_count, 0)
        self.assertIsNone(report.node_stats["first_buy"].hit_rate_60d)


def _cyclical_candles(count: int) -> list[MarketCandle]:
    candles: list[MarketCandle] = []
    for index in range(count):
        cycle = index % 80
        if cycle < 40:
            close = 100 + cycle
        elif cycle < 58:
            close = 140 - (cycle - 40) * 1.8
        else:
            close = 108 + (cycle - 58) * 1.2
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


if __name__ == "__main__":
    unittest.main()
