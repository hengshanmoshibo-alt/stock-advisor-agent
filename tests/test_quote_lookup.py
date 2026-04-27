from __future__ import annotations

import asyncio
import unittest

from invest_digital_human.quote_lookup import (
    QuoteSnapshot,
    build_node_distances,
    build_quote_distance_summary,
    lookup_quote,
    parse_node_prices,
    quote_lookup_requested,
    ticker_for_source,
)
from invest_digital_human.stock_nodes import StockNodeRecord


class QuoteLookupTest(unittest.TestCase):
    def test_ticker_mapping(self) -> None:
        self.assertEqual(ticker_for_source("nvda"), "NVDA")
        self.assertEqual(ticker_for_source("brk"), "BRK-B")
        self.assertEqual(ticker_for_source("costco"), "COST")
        self.assertEqual(ticker_for_source("v"), "V")

    def test_parse_node_prices(self) -> None:
        self.assertEqual(parse_node_prices(["170以内", "169", "90-95", "85左右"]), [170, 169, 90, 95, 85])

    def test_build_quote_distance_summary(self) -> None:
        quote = QuoteSnapshot(source_key="nvda", ticker="NVDA", price=180, previous_close=175)
        records = [
            StockNodeRecord(
                stock="NVDA",
                display_stock="英伟达",
                source_key="nvda",
                has_explicit_nodes=True,
                date="2026-04-14",
                article_file="0004_2026-04-14_test.html",
                entry_type="加仓节点",
                nodes=["170以内", "169", "165"],
                summary="",
                evidence="",
            )
        ]

        distances = build_node_distances(quote, records)
        self.assertAlmostEqual(distances[0].distance_percent, 5.8823, places=3)
        summary = build_quote_distance_summary(quote, records)
        self.assertIsNotNone(summary)
        self.assertIn("NVDA", summary or "")
        self.assertIn("180.00", summary or "")
        self.assertIn("2026-04-14", summary or "")

    def test_quote_lookup_intent(self) -> None:
        self.assertTrue(quote_lookup_requested("AMD 现在离买点还有多远？"))
        self.assertTrue(quote_lookup_requested("NVDA current price distance"))
        self.assertFalse(quote_lookup_requested("AMD 的买入节点有哪些？"))

    def test_finnhub_lookup_uses_quote_endpoint(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, float]:
                return {"c": 150.25, "pc": 145.0, "t": 1710000000}

        class FakeClient:
            def __init__(self) -> None:
                self.url = ""
                self.params: dict[str, str] = {}

            async def get(self, url: str, params: dict[str, str]) -> FakeResponse:
                self.url = url
                self.params = params
                return FakeResponse()

        client = FakeClient()
        quote = asyncio.run(
            lookup_quote(
                "amd",
                provider="finnhub",
                api_key="test-token",
                client=client,  # type: ignore[arg-type]
            )
        )

        self.assertIsNotNone(quote)
        self.assertEqual(quote.provider, "finnhub")  # type: ignore[union-attr]
        self.assertEqual(quote.ticker, "AMD")  # type: ignore[union-attr]
        self.assertEqual(client.url, "https://finnhub.io/api/v1/quote")
        self.assertEqual(client.params["symbol"], "AMD")
        self.assertEqual(client.params["token"], "test-token")


if __name__ == "__main__":
    unittest.main()
