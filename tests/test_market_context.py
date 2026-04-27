from __future__ import annotations

import asyncio
import unittest

from invest_digital_human.market_context import build_market_context
from invest_digital_human.market_data import MarketCandle


class MarketContextTest(unittest.TestCase):
    def test_build_market_context_flags_risk_off(self) -> None:
        context = asyncio.run(build_market_context(FakeMarketData(), extra_tickers=["XLK", "QQQ"]))

        self.assertIsNotNone(context)
        self.assertTrue(context.risk_off)  # type: ignore[union-attr]
        facts = context.as_facts()  # type: ignore[union-attr]
        self.assertEqual(len(facts["indices"]), 3)
        self.assertEqual([item["ticker"] for item in facts["indices"]], ["QQQ", "SPY", "XLK"])


class FakeMarketData:
    provider = "massive"

    async def get_daily_candles(self, source_key: str, *, lookback_days: int = 420) -> list[MarketCandle]:
        if source_key == "qqq":
            return _downtrend_candles(220)
        return _uptrend_candles(220)


def _uptrend_candles(count: int) -> list[MarketCandle]:
    return [
        MarketCandle(
            timestamp=1_700_000_000 + index * 86_400,
            date=f"2025-01-{(index % 28) + 1:02d}",
            open=99 + index,
            high=101 + index,
            low=99 + index,
            close=100 + index,
            volume=1_000_000,
        )
        for index in range(count)
    ]


def _downtrend_candles(count: int) -> list[MarketCandle]:
    return [
        MarketCandle(
            timestamp=1_700_000_000 + index * 86_400,
            date=f"2025-01-{(index % 28) + 1:02d}",
            open=500 - index,
            high=501 - index,
            low=499 - index,
            close=500 - index,
            volume=1_000_000,
        )
        for index in range(count)
    ]


if __name__ == "__main__":
    unittest.main()
