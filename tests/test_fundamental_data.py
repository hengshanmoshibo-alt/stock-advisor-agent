from __future__ import annotations

import unittest

from invest_digital_human.fundamental_data import (
    _parse_latest_earnings,
    _parse_latest_recommendation,
    _parse_next_earnings,
    _pick_metrics,
)


class FundamentalDataTest(unittest.TestCase):
    def test_pick_metrics_keeps_supported_fields(self) -> None:
        metrics = _pick_metrics(
            {
                "52WeekHigh": 555.45,
                "52WeekLow": 356.28,
                "peNormalizedAnnual": 29.5,
                "ignored": 1,
            }
        )

        self.assertEqual(metrics["52WeekHigh"], 555.45)
        self.assertEqual(metrics["52WeekLow"], 356.28)
        self.assertEqual(metrics["peNormalizedAnnual"], 29.5)
        self.assertNotIn("ignored", metrics)

    def test_parse_latest_earnings(self) -> None:
        item = _parse_latest_earnings(
            [{"period": "2025-12-31", "actual": 4.14, "estimate": 4.03, "surprisePercent": 2.61}]
        )

        self.assertIsNotNone(item)
        self.assertEqual(item.period, "2025-12-31")  # type: ignore[union-attr]
        self.assertAlmostEqual(item.surprise_percent or 0, 2.61)  # type: ignore[union-attr]

    def test_parse_recommendation(self) -> None:
        item = _parse_latest_recommendation(
            [{"period": "2026-04-01", "strongBuy": 23, "buy": 36, "hold": 6, "sell": 0, "strongSell": 0}]
        )

        self.assertIsNotNone(item)
        self.assertEqual(item.positive_count, 59)  # type: ignore[union-attr]
        self.assertEqual(item.negative_count, 0)  # type: ignore[union-attr]

    def test_parse_next_earnings_sorts_by_date(self) -> None:
        item = _parse_next_earnings(
            {
                "earningsCalendar": [
                    {"date": "2026-07-28", "epsEstimate": 4.3, "revenueEstimate": 89},
                    {"date": "2026-04-29", "epsEstimate": 4.1, "revenueEstimate": 80},
                ]
            }
        )

        self.assertIsNotNone(item)
        self.assertEqual(item.date, "2026-04-29")  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
