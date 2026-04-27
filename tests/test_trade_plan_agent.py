from __future__ import annotations

import unittest

from invest_digital_human.fundamental_data import EarningsEvent, FundamentalSnapshot
from invest_digital_human.market_context import IndexContext, MarketContext
from invest_digital_human.market_data import MarketCandle
from invest_digital_human.quote_lookup import QuoteSnapshot
from invest_digital_human.trade_plan_agent import build_trade_plan


class TradePlanAgentTest(unittest.TestCase):
    def test_builds_wait_plan_for_weak_state_and_earnings_risk(self) -> None:
        result = build_trade_plan(
            display_stock="MSFT",
            quote=QuoteSnapshot(source_key="msft", ticker="MSFT", price=150),
            candles=_candles(220),
            fundamentals=FundamentalSnapshot(
                ticker="MSFT",
                metrics={},
                latest_earnings=None,
                next_earnings=EarningsEvent(
                    date="2026-04-29",
                    eps_estimate=4.1,
                    revenue_estimate=80_000,
                ),
                latest_recommendation=None,
            ),
            market_context=MarketContext(
                indices=[
                    IndexContext(
                        ticker="QQQ",
                        current_price=300,
                        ma50=320,
                        ma200=330,
                        trend="below_ma200",
                    )
                ]
            ),
        )

        self.assertEqual(result.action_state, "wait")
        self.assertEqual(result.confidence, "low")
        self.assertIn("现在怎么做", result.answer)
        self.assertIn("关键价格", result.answer)
        self.assertIn("触发条件", result.answer)
        self.assertIn("风险预案", result.answer)
        self.assertNotIn("Historical article nodes", result.answer)
        self.assertIsNotNone(result.trade_plan)
        assert result.trade_plan is not None
        self.assertEqual(result.trade_plan.ticker, "MSFT")
        self.assertEqual(result.trade_plan.action_state, "wait")
        self.assertEqual(result.trade_plan.confidence, "low")
        self.assertEqual(result.trade_plan.current_price, 150)
        self.assertTrue(any(node.key == "observation" for node in result.trade_plan.nodes))
        self.assertTrue(result.trade_plan.confirmation_condition)
        self.assertTrue(result.trade_plan.failure_condition)
        self.assertIsNotNone(result.trade_plan.volume)
        self.assertIsNotNone(result.trade_plan.backtest)
        self.assertIsNotNone(result.trade_plan.score_breakdown)
        self.assertTrue(result.trade_plan.trend_stage)
        self.assertEqual(result.trade_plan.parameter_source, "default_formula")
        self.assertIsNotNone(result.trade_plan.calibration)
        self.assertIsNotNone(result.trade_plan.backtest_summary)
        assert result.trade_plan.score_breakdown is not None
        self.assertLess(result.trade_plan.score_breakdown.total, 0)

    def test_wait_plan_explains_defense_as_future_risk_plan(self) -> None:
        result = build_trade_plan(
            display_stock="AMZN",
            quote=QuoteSnapshot(source_key="amzn", ticker="AMZN", price=319),
            candles=_candles(220),
            fundamentals=None,
            market_context=None,
        )

        self.assertIn("风控预案", result.answer)
        self.assertIn("不是当前买点", result.answer)
        self.assertIn("企稳确认", result.answer)
        self.assertLess(len(result.answer), 800)
        self.assertIsNotNone(result.trade_plan)
        assert result.trade_plan is not None
        defense = next(node for node in result.trade_plan.nodes if node.key == "defense")
        first_buy = next(node for node in result.trade_plan.nodes if node.key == "first_buy")
        self.assertEqual(defense.role_label, "买入后风控")
        self.assertIn("不是买入点", defense.plain_explanation or "")
        self.assertEqual(first_buy.role_label, "分批候选")

    def test_missing_tools_do_not_fabricate_nodes(self) -> None:
        result = build_trade_plan(
            display_stock="MSFT",
            quote=None,
            candles=[],
            fundamentals=None,
            market_context=None,
        )

        self.assertEqual(result.action_state, "wait")
        self.assertEqual(result.confidence, "low")
        self.assertIn("不会在数据不足时编造买入节点", result.answer)
        self.assertIsNone(result.trade_plan)


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


if __name__ == "__main__":
    unittest.main()
