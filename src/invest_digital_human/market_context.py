from __future__ import annotations

from dataclasses import dataclass

from .market_data import MarketDataClient
from .technical_nodes import TechnicalSnapshot, calculate_technical_node_plan
from .quote_lookup import QuoteSnapshot


@dataclass(frozen=True, slots=True)
class IndexContext:
    ticker: str
    current_price: float
    ma50: float | None
    ma200: float | None
    trend: str

    def as_facts(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "current_price": round(self.current_price, 2),
            "ma50": round(self.ma50, 2) if self.ma50 is not None else None,
            "ma200": round(self.ma200, 2) if self.ma200 is not None else None,
            "trend": self.trend,
        }


@dataclass(frozen=True, slots=True)
class MarketContext:
    indices: list[IndexContext]

    @property
    def risk_off(self) -> bool:
        return any(item.trend == "below_ma200" for item in self.indices)

    def as_facts(self) -> dict[str, object]:
        return {
            "risk_off": self.risk_off,
            "indices": [item.as_facts() for item in self.indices],
        }


async def build_market_context(
    market_data: MarketDataClient,
    *,
    extra_tickers: list[str] | tuple[str, ...] = (),
) -> MarketContext | None:
    indices: list[IndexContext] = []
    tickers = []
    for ticker in ("qqq", "spy", *extra_tickers):
        normalized = ticker.strip().lower()
        if normalized and normalized not in tickers:
            tickers.append(normalized)
    for ticker in tickers:
        candles = await market_data.get_daily_candles(ticker, lookback_days=420)
        if not candles:
            continue
        quote = QuoteSnapshot(
            source_key=ticker,
            ticker=ticker.upper(),
            price=candles[-1].close,
            provider=market_data.provider,
        )
        plan = calculate_technical_node_plan(candles=candles, quote=quote)
        if plan is None:
            continue
        snapshot: TechnicalSnapshot = plan.snapshot
        if snapshot.ma200 is not None and snapshot.current_price < snapshot.ma200:
            trend = "below_ma200"
        elif snapshot.ma50 is not None and snapshot.current_price < snapshot.ma50:
            trend = "below_ma50"
        else:
            trend = "above_ma50"
        indices.append(
            IndexContext(
                ticker=quote.ticker,
                current_price=snapshot.current_price,
                ma50=snapshot.ma50,
                ma200=snapshot.ma200,
                trend=trend,
            )
        )
    return MarketContext(indices=indices) if indices else None
