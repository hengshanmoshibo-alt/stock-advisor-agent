from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from .quote_lookup import ticker_for_source


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    date: str
    eps_estimate: float | None
    revenue_estimate: float | None
    hour: str | None = None

    def as_facts(self) -> dict[str, object]:
        return {
            "date": self.date,
            "eps_estimate": self.eps_estimate,
            "revenue_estimate": self.revenue_estimate,
            "hour": self.hour,
        }


@dataclass(frozen=True, slots=True)
class EarningsSurprise:
    period: str
    actual: float | None
    estimate: float | None
    surprise_percent: float | None

    def as_facts(self) -> dict[str, object]:
        return {
            "period": self.period,
            "actual": self.actual,
            "estimate": self.estimate,
            "surprise_percent": self.surprise_percent,
        }


@dataclass(frozen=True, slots=True)
class RecommendationTrend:
    period: str
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int

    @property
    def positive_count(self) -> int:
        return self.strong_buy + self.buy

    @property
    def negative_count(self) -> int:
        return self.sell + self.strong_sell

    def as_facts(self) -> dict[str, object]:
        return {
            "period": self.period,
            "strong_buy": self.strong_buy,
            "buy": self.buy,
            "hold": self.hold,
            "sell": self.sell,
            "strong_sell": self.strong_sell,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


@dataclass(frozen=True, slots=True)
class FundamentalSnapshot:
    ticker: str
    metrics: dict[str, float]
    latest_earnings: EarningsSurprise | None
    next_earnings: EarningsEvent | None
    latest_recommendation: RecommendationTrend | None

    def days_to_earnings(self, *, today: date | None = None) -> int | None:
        if self.next_earnings is None:
            return None
        try:
            event_date = date.fromisoformat(self.next_earnings.date)
        except ValueError:
            return None
        return (event_date - (today or date.today())).days

    def as_facts(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "metrics": self.metrics,
            "latest_earnings": self.latest_earnings.as_facts() if self.latest_earnings else None,
            "next_earnings": self.next_earnings.as_facts() if self.next_earnings else None,
            "latest_recommendation": (
                self.latest_recommendation.as_facts()
                if self.latest_recommendation
                else None
            ),
        }


class FundamentalDataClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        timeout: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.client = client
        self._cache: dict[str, tuple[datetime, FundamentalSnapshot | None]] = {}

    async def get_snapshot(self, source_key: str) -> FundamentalSnapshot | None:
        if not self.api_key:
            return None
        ticker = ticker_for_source(source_key)
        cached = self._cache.get(ticker)
        if cached is not None and datetime.now(tz=timezone.utc) - cached[0] < timedelta(hours=1):
            return cached[1]
        close_client = self.client is None
        http_client = self.client or httpx.AsyncClient(timeout=self.timeout)
        try:
            metric_payload = await self._get_json(
                http_client,
                "https://finnhub.io/api/v1/stock/metric",
                {"symbol": ticker, "metric": "all", "token": self.api_key},
            )
            earnings_payload = await self._get_json(
                http_client,
                "https://finnhub.io/api/v1/stock/earnings",
                {"symbol": ticker, "token": self.api_key},
            )
            recommendation_payload = await self._get_json(
                http_client,
                "https://finnhub.io/api/v1/stock/recommendation",
                {"symbol": ticker, "token": self.api_key},
            )
            calendar_payload = await self._get_json(
                http_client,
                "https://finnhub.io/api/v1/calendar/earnings",
                {
                    "symbol": ticker,
                    "from": date.today().isoformat(),
                    "to": date(date.today().year + 1, date.today().month, date.today().day).isoformat(),
                    "token": self.api_key,
                },
            )
            snapshot = FundamentalSnapshot(
                ticker=ticker,
                metrics=_pick_metrics(metric_payload.get("metric") if metric_payload else {}),
                latest_earnings=_parse_latest_earnings(earnings_payload),
                next_earnings=_parse_next_earnings(calendar_payload),
                latest_recommendation=_parse_latest_recommendation(recommendation_payload),
            )
            self._cache[ticker] = (datetime.now(tz=timezone.utc), snapshot)
            return snapshot
        except (httpx.HTTPError, TypeError, ValueError):
            self._cache[ticker] = (datetime.now(tz=timezone.utc), None)
            return None
        finally:
            if close_client:
                await http_client.aclose()

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str],
    ) -> dict[str, Any] | list[Any] | None:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _pick_metrics(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    wanted = [
        "52WeekHigh",
        "52WeekLow",
        "52WeekPriceReturnDaily",
        "10DayAverageTradingVolume",
        "3MonthAverageTradingVolume",
        "peBasicExclExtraTTM",
        "peNormalizedAnnual",
        "psTTM",
        "revenueGrowthTTMYoy",
        "epsGrowthTTMYoy",
    ]
    metrics: dict[str, float] = {}
    for key in wanted:
        value = _first_float(payload.get(key))
        if value is not None:
            metrics[key] = value
    return metrics


def _parse_latest_earnings(payload: object) -> EarningsSurprise | None:
    if not isinstance(payload, list) or not payload:
        return None
    row = payload[0]
    if not isinstance(row, dict):
        return None
    return EarningsSurprise(
        period=str(row.get("period") or ""),
        actual=_first_float(row.get("actual")),
        estimate=_first_float(row.get("estimate")),
        surprise_percent=_first_float(row.get("surprisePercent")),
    )


def _parse_latest_recommendation(payload: object) -> RecommendationTrend | None:
    if not isinstance(payload, list) or not payload:
        return None
    row = payload[0]
    if not isinstance(row, dict):
        return None
    return RecommendationTrend(
        period=str(row.get("period") or ""),
        strong_buy=_first_int(row.get("strongBuy")),
        buy=_first_int(row.get("buy")),
        hold=_first_int(row.get("hold")),
        sell=_first_int(row.get("sell")),
        strong_sell=_first_int(row.get("strongSell")),
    )


def _parse_next_earnings(payload: object) -> EarningsEvent | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get("earningsCalendar")
    if not isinstance(rows, list) or not rows:
        return None
    parsed_rows = [row for row in rows if isinstance(row, dict) and row.get("date")]
    parsed_rows.sort(key=lambda row: str(row.get("date")))
    if not parsed_rows:
        return None
    row = parsed_rows[0]
    return EarningsEvent(
        date=str(row.get("date")),
        eps_estimate=_first_float(row.get("epsEstimate")),
        revenue_estimate=_first_float(row.get("revenueEstimate")),
        hour=str(row.get("hour")) if row.get("hour") is not None else None,
    )


def _first_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
