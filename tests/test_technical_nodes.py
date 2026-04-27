from __future__ import annotations

import unittest

from invest_digital_human.backtest_engine import StrategyParameters, backtest_node_strategy
from invest_digital_human.calibration import TradeNodeCalibration
from invest_digital_human.market_data import MarketCandle
from invest_digital_human.fundamental_data import (
    EarningsEvent,
    EarningsSurprise,
    FundamentalSnapshot,
    RecommendationTrend,
)
from invest_digital_human.market_context import IndexContext, MarketContext
from invest_digital_human.quote_lookup import QuoteSnapshot
from invest_digital_human.technical_nodes import (
    calculated_nodes_requested,
    calculate_technical_node_plan,
    format_enhanced_calculated_node_answer,
    format_calculated_node_answer,
)


class TechnicalNodesTest(unittest.TestCase):
    def test_calculates_uptrend_nodes(self) -> None:
        candles = _candles(220)
        quote = QuoteSnapshot(source_key="amd", ticker="AMD", price=319)

        plan = calculate_technical_node_plan(candles=candles, quote=quote)

        self.assertIsNotNone(plan)
        self.assertTrue(plan.data_sufficient)  # type: ignore[union-attr]
        snapshot = plan.snapshot  # type: ignore[union-attr]
        self.assertAlmostEqual(snapshot.ma20 or 0, 309.5)
        self.assertAlmostEqual(snapshot.ma50 or 0, 294.5)
        self.assertAlmostEqual(snapshot.ma200 or 0, 219.5)
        self.assertAlmostEqual(snapshot.atr14 or 0, 2.0)
        self.assertAlmostEqual(snapshot.rsi14 or 0, 100.0)
        nodes = {node.key: node for node in plan.nodes}  # type: ignore[union-attr]
        self.assertAlmostEqual(nodes["observation"].lower, 292.5)
        self.assertAlmostEqual(nodes["observation"].upper, 294.5)
        self.assertAlmostEqual(nodes["first_buy"].lower, 256.0)
        self.assertAlmostEqual(nodes["first_buy"].upper, 272.0)
        self.assertIn("Defense", nodes["defense"].title)
        self.assertEqual(plan.trend_stage, "extended_overheated")  # type: ignore[union-attr]
        self.assertEqual(plan.volume_profile.signal, "neutral")  # type: ignore[union-attr]
        self.assertGreaterEqual(plan.backtest_calibration.sample_count, 0)  # type: ignore[union-attr]
        self.assertEqual(plan.parameter_source, "default_formula")  # type: ignore[union-attr]
        self.assertIsNotNone(plan.backtest_summary)  # type: ignore[union-attr]

    def test_calibration_changes_first_buy_zone(self) -> None:
        candles = _calibration_candles(420)
        quote = QuoteSnapshot(source_key="amd", ticker="AMD", price=450)
        params = StrategyParameters(first_buy_drawdown_band=(0.05, 0.10), min_sample_count=0)
        calibration = TradeNodeCalibration(
            source_key="amd",
            generated_at="2026-01-01T00:00:00+00:00",
            parameters=params,
            backtest_report=backtest_node_strategy(candles, parameters=params, parameter_source="calibrated"),
        )

        default_plan = calculate_technical_node_plan(candles=candles, quote=quote)
        calibrated_plan = calculate_technical_node_plan(candles=candles, quote=quote, calibration=calibration)

        self.assertIsNotNone(default_plan)
        self.assertIsNotNone(calibrated_plan)
        self.assertEqual(calibrated_plan.parameter_source, "calibrated")  # type: ignore[union-attr]
        default_nodes = {node.key: node for node in default_plan.nodes}  # type: ignore[union-attr]
        calibrated_nodes = {node.key: node for node in calibrated_plan.nodes}  # type: ignore[union-attr]
        self.assertNotEqual(
            default_nodes["first_buy"].lower,
            calibrated_nodes["first_buy"].lower,
        )

    def test_weak_state_only_outputs_observation(self) -> None:
        candles = _candles(220)
        quote = QuoteSnapshot(source_key="amd", ticker="AMD", price=150)

        plan = calculate_technical_node_plan(candles=candles, quote=quote)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.risk_state, "weak")  # type: ignore[union-attr]
        self.assertEqual([node.key for node in plan.nodes], ["observation"])  # type: ignore[union-attr]

    def test_observation_action_matches_current_price_location(self) -> None:
        plan = calculate_technical_node_plan(
            candles=_candles(220),
            quote=QuoteSnapshot(source_key="amd", ticker="AMD", price=293),
        )

        self.assertIsNotNone(plan)
        nodes = {node.key: node for node in plan.nodes}  # type: ignore[union-attr]
        self.assertIn("已进入观察区", nodes["observation"].action)
        self.assertNotIn("价格进入该区域后", nodes["observation"].action)

    def test_insufficient_data_is_not_sufficient(self) -> None:
        plan = calculate_technical_node_plan(
            candles=_candles(30),
            quote=QuoteSnapshot(source_key="amd", ticker="AMD", price=130),
        )

        self.assertIsNotNone(plan)
        self.assertFalse(plan.data_sufficient)  # type: ignore[union-attr]
        self.assertEqual(plan.nodes, [])  # type: ignore[union-attr]

    def test_request_detection_and_formatting(self) -> None:
        self.assertTrue(calculated_nodes_requested("AMD 现在买入节点是多少？"))
        self.assertTrue(calculated_nodes_requested("NVDA technical entry"))
        self.assertFalse(calculated_nodes_requested("AMD 的买入节点有哪些？"))

        plan = calculate_technical_node_plan(
            candles=_candles(220),
            quote=QuoteSnapshot(source_key="amd", ticker="AMD", price=319),
        )
        answer = format_calculated_node_answer(plan)  # type: ignore[arg-type]
        self.assertIn("Calculated nodes", answer)
        self.assertIn("Observation zone", answer)
        self.assertIn("第一买入区", answer)

    def test_enhanced_answer_includes_fundamentals_and_market_context(self) -> None:
        plan = calculate_technical_node_plan(
            candles=_candles(220),
            quote=QuoteSnapshot(source_key="amd", ticker="AMD", price=319),
        )
        fundamentals = FundamentalSnapshot(
            ticker="AMD",
            metrics={"52WeekHigh": 320, "52WeekLow": 100, "peNormalizedAnnual": 30},
            latest_earnings=EarningsSurprise(
                period="2026-01-01",
                actual=1.2,
                estimate=1.0,
                surprise_percent=20,
            ),
            next_earnings=EarningsEvent(
                date="2026-05-01",
                eps_estimate=1.4,
                revenue_estimate=10_000,
            ),
            latest_recommendation=RecommendationTrend(
                period="2026-04-01",
                strong_buy=10,
                buy=20,
                hold=5,
                sell=1,
                strong_sell=0,
            ),
        )
        market = MarketContext(
            indices=[
                IndexContext(
                    ticker="QQQ",
                    current_price=400,
                    ma50=420,
                    ma200=430,
                    trend="below_ma200",
                )
            ]
        )

        answer = format_enhanced_calculated_node_answer(
            plan,  # type: ignore[arg-type]
            fundamentals=fundamentals,
            market_context=market,
        )

        self.assertIn("Additional confirmation data", answer)
        self.assertIn("Fundamental/earnings context", answer)
        self.assertIn("Analyst trend", answer)
        self.assertIn("Market risk", answer)


def _candles(count: int) -> list[MarketCandle]:
    return [
        MarketCandle(
            timestamp=1_700_000_000 + index * 86_400,
            date=f"2025-01-{(index % 28) + 1:02d}",
            open=99 + index,
            high=101 + index,
            low=99 + index,
            close=100 + index,
            volume=1_000_000 + index,
        )
        for index in range(count)
    ]


def _calibration_candles(count: int) -> list[MarketCandle]:
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
