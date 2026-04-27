from __future__ import annotations

import unittest
from pathlib import Path

from invest_digital_human.agent_planner import AgentPlanner
from invest_digital_human.stock_nodes import StockNodeKnowledgeBase


class AgentPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = AgentPlanner(
            StockNodeKnowledgeBase(Path("data/final_finetune/buy_nodes_master.jsonl"))
        )

    def test_routes_current_buy_question_to_trade_plan(self) -> None:
        plan = self.planner.plan("微软现在买入节点是多少？")

        self.assertEqual(plan.intent, "trade_plan")
        self.assertEqual(plan.source_key, "msft")
        self.assertFalse(plan.use_articles)
        self.assertIn("quote_lookup", plan.tools)
        self.assertIn("market_candles", plan.tools)

    def test_routes_history_question_to_stock_nodes(self) -> None:
        plan = self.planner.plan("微软历史文章里买点有哪些？")

        self.assertEqual(plan.intent, "stock_nodes")
        self.assertEqual(plan.source_key, "msft")
        self.assertTrue(plan.use_articles)
        self.assertEqual(plan.tools, ("article_rag",))

    def test_routes_generic_question_to_generic_rag(self) -> None:
        plan = self.planner.plan("怎么看最近美股回调？")

        self.assertEqual(plan.intent, "market_explain")


if __name__ == "__main__":
    unittest.main()
