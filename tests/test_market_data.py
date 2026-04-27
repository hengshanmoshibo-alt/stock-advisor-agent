from __future__ import annotations

import asyncio
import unittest

from invest_digital_human.market_data import MarketDataClient, parse_finnhub_candles, parse_massive_candles


class MarketDataTest(unittest.TestCase):
    def test_parse_finnhub_candles_sorts_rows(self) -> None:
        payload = {
            "s": "ok",
            "t": [300, 100, 200],
            "o": [3, 1, 2],
            "h": [4, 2, 3],
            "l": [2, 0.5, 1.5],
            "c": [3.5, 1.5, 2.5],
            "v": [30, 10, 20],
        }

        candles = parse_finnhub_candles(payload)

        self.assertEqual([item.timestamp for item in candles], [100, 200, 300])
        self.assertEqual(candles[0].open, 1)
        self.assertEqual(candles[-1].close, 3.5)

    def test_market_client_returns_empty_without_key(self) -> None:
        client = MarketDataClient(provider="finnhub", api_key=None)

        candles = asyncio.run(client.get_daily_candles("amd"))

        self.assertEqual(candles, [])

    def test_market_client_uses_finnhub_candle_endpoint(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "s": "ok",
                    "t": [100],
                    "o": [1],
                    "h": [2],
                    "l": [0.5],
                    "c": [1.5],
                    "v": [1000],
                }

        class FakeClient:
            def __init__(self) -> None:
                self.url = ""
                self.params: dict[str, str] = {}

            async def get(self, url: str, params: dict[str, str]) -> FakeResponse:
                self.url = url
                self.params = params
                return FakeResponse()

        fake = FakeClient()
        client = MarketDataClient(provider="finnhub", api_key="token", client=fake)  # type: ignore[arg-type]

        candles = asyncio.run(client.get_daily_candles("amd"))

        self.assertEqual(len(candles), 1)
        self.assertEqual(fake.url, "https://finnhub.io/api/v1/stock/candle")
        self.assertEqual(fake.params["symbol"], "AMD")
        self.assertEqual(fake.params["resolution"], "D")
        self.assertEqual(fake.params["token"], "token")

    def test_parse_massive_candles_sorts_rows(self) -> None:
        payload = {
            "results": [
                {"t": 200_000, "o": 2, "h": 3, "l": 1, "c": 2.5, "v": 20},
                {"t": 100_000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            ]
        }

        candles = parse_massive_candles(payload)

        self.assertEqual([item.timestamp for item in candles], [100, 200])
        self.assertEqual(candles[0].close, 1.5)
        self.assertEqual(candles[-1].volume, 20)

    def test_market_client_uses_massive_aggregate_endpoint(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "results": [
                        {"t": 100_000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1000},
                    ]
                }

        class FakeClient:
            def __init__(self) -> None:
                self.url = ""
                self.params: dict[str, str] = {}

            async def get(self, url: str, params: dict[str, str]) -> FakeResponse:
                self.url = url
                self.params = params
                return FakeResponse()

        fake = FakeClient()
        client = MarketDataClient(provider="massive", api_key="token", client=fake)  # type: ignore[arg-type]

        candles = asyncio.run(client.get_daily_candles("amd"))

        self.assertEqual(len(candles), 1)
        self.assertIn("https://api.massive.com/v2/aggs/ticker/AMD/range/1/day/", fake.url)
        self.assertEqual(fake.params["adjusted"], "true")
        self.assertEqual(fake.params["sort"], "asc")
        self.assertEqual(fake.params["apiKey"], "token")


if __name__ == "__main__":
    unittest.main()
