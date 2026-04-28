from __future__ import annotations

import asyncio
import os
import unittest

from invest_digital_human.config import AppSettings
from invest_digital_human.conversation_intent_resolver import (
    ConversationIntent,
    ConversationIntentResolver,
)
from invest_digital_human.fundamental_data import FundamentalSnapshot
from invest_digital_human.market_data import MarketCandle
from invest_digital_human.quote_lookup import QuoteSnapshot
from invest_digital_human.schemas import ChatPayload, SessionState, TradePlanMetrics, TradePlanNode, TradePlanPayload
from invest_digital_human.stock_advisor_explainer import CONCEPT_DISCLAIMER
from invest_digital_human.stock_advisor_service import StockAdvisorService


class StockAdvisorServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["INDEX_PATH"] = "data/index.json"
        os.environ["RERANKER_STRATEGY"] = "none"
        os.environ["OLLAMA_MODEL"] = "none"
        os.environ["API_KEY"] = ""
        os.environ["BASE_URL"] = ""
        os.environ["MODEL"] = ""
        os.environ["INVEST_DH_API_KEY"] = ""
        os.environ["INVEST_DH_BASE_URL"] = ""
        os.environ["INVEST_DH_MODEL"] = ""
        os.environ["TRADE_NODE_CALIBRATION_PATH"] = "data/test_missing_calibration.json"

    def test_concept_question_uses_explainer(self) -> None:
        service = StockAdvisorService()
        service.intent_resolver = FakeIntentResolver(
            [ConversationIntent(intent="concept_explain", relation_to_last_trade="none", confidence=0.95)]
        )
        service.explainer = FakeExplainer("MA200 是 200 日均线。")
        session = service.sessions.create()

        payload = asyncio.run(service._build_payload(session=session, message="MA200 是什么？"))

        self.assertEqual(payload.answer_mode, "stock_advisor_explainer")
        self.assertIsNone(payload.trade_plan)
        self.assertEqual(payload.disclaimer, CONCEPT_DISCLAIMER)

    def test_bare_buy_node_returns_clarification(self) -> None:
        service = StockAdvisorService()
        service.intent_resolver = FakeIntentResolver(
            [
                ConversationIntent(
                    intent="clarify",
                    relation_to_last_trade="ambiguous",
                    confidence=0.9,
                    clarifying_question="请先说明具体股票。",
                )
            ]
        )
        session = service.sessions.create()

        payload = asyncio.run(service._build_payload(session=session, message="买入节点"))

        self.assertEqual(payload.answer_mode, "stock_advisor_clarify")
        self.assertIsNone(payload.trade_plan)
        self.assertIn("具体股票", payload.answer)

    def test_ui_session_history_records_messages_and_title(self) -> None:
        service = StockAdvisorService()
        session = service.sessions.create()
        payload = make_chat_payload(session.session_id, answer="我是助手回答")

        service._record_ui_history(session=session, user_message="我是谁呢？", payload=payload)

        sessions = service.list_sessions()
        history = service.get_session_history(session.session_id)
        self.assertEqual(sessions[0]["session_id"], session.session_id)
        self.assertEqual(sessions[0]["title"], "我是谁呢？")
        self.assertEqual(len(history["messages"]), 2)
        self.assertEqual(history["messages"][0]["role"], "user")
        self.assertEqual(history["messages"][1]["role"], "assistant")
        self.assertEqual(history["messages"][1]["text"], "我是助手回答")

    def test_ui_session_history_keeps_one_conversation_for_multiple_turns(self) -> None:
        service = StockAdvisorService()
        session = service.sessions.create()

        service._record_ui_history(
            session=session,
            user_message="我是谁呢？",
            payload=make_chat_payload(session.session_id, answer="第一轮回答"),
        )
        service._record_ui_history(
            session=session,
            user_message="你是谁呢？",
            payload=make_chat_payload(session.session_id, answer="第二轮回答"),
        )

        sessions = service.list_sessions()
        history = service.get_session_history(session.session_id)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["title"], "我是谁呢？")
        self.assertEqual(len(history["messages"]), 4)

    def test_trade_plan_followup_uses_llm_intent_and_previous_plan(self) -> None:
        service = StockAdvisorService()
        service.intent_resolver = FakeIntentResolver(
            [
                ConversationIntent(
                    intent="trade_plan",
                    relation_to_last_trade="none",
                    target_ticker="NVDA",
                    target_name="英伟达",
                    confidence=0.95,
                ),
                ConversationIntent(
                    intent="context_followup",
                    relation_to_last_trade="followup",
                    target_ticker="NVDA",
                    target_name="英伟达",
                    requested_field="first_buy_zone",
                    confidence=0.95,
                ),
            ]
        )
        service.quote_lookup = fake_quote
        service.market_data = FakeMarketData()
        service.fundamentals = FakeFundamentals()
        session = service.sessions.create()

        first = asyncio.run(service._build_payload(session=session, message="英伟达现在能不能买？"))
        second = asyncio.run(service._build_payload(session=session, message="那第一买点是多少？"))

        self.assertIsNotNone(first.trade_plan)
        self.assertEqual(second.answer_mode, "trade_plan_context")
        self.assertIsNone(second.trade_plan)
        self.assertIn("英伟达", second.answer)
        self.assertIn("第一买入区", second.answer)

    def test_trade_plan_rationale_followup_explains_why_not_buy_now(self) -> None:
        service = StockAdvisorService()
        service.intent_resolver = FakeIntentResolver(
            [
                ConversationIntent(
                    intent="trade_plan",
                    relation_to_last_trade="none",
                    target_ticker="NVDA",
                    target_name="英伟达",
                    confidence=0.95,
                ),
                ConversationIntent(
                    intent="context_followup",
                    relation_to_last_trade="followup",
                    target_ticker="NVDA",
                    target_name="英伟达",
                    requested_field="rationale",
                    confidence=0.95,
                ),
            ]
        )
        service.quote_lookup = fake_quote
        service.market_data = FakeMarketData()
        service.fundamentals = FakeFundamentals()
        session = service.sessions.create()

        asyncio.run(service._build_payload(session=session, message="英伟达现在能不能买？"))
        payload = asyncio.run(service._build_payload(session=session, message="为什么不是现在买？"))

        self.assertEqual(payload.answer_mode, "trade_plan_context")
        self.assertIn("不直接现在买", payload.answer)
        self.assertIn("计划等级", payload.answer)
        self.assertNotIn("medium", payload.answer)
        self.assertIn("确认条件", payload.answer)

    def test_trade_plan_defense_followup_explains_risk_line(self) -> None:
        service = StockAdvisorService()
        service.intent_resolver = FakeIntentResolver(
            [
                ConversationIntent(
                    intent="trade_plan",
                    relation_to_last_trade="none",
                    target_ticker="NVDA",
                    target_name="英伟达",
                    confidence=0.95,
                ),
                ConversationIntent(
                    intent="context_followup",
                    relation_to_last_trade="followup",
                    target_ticker="NVDA",
                    target_name="英伟达",
                    requested_field="defense",
                    confidence=0.95,
                ),
            ]
        )
        service.quote_lookup = fake_quote
        service.market_data = FakeMarketData()
        service.fundamentals = FakeFundamentals()
        session = service.sessions.create()

        asyncio.run(service._build_payload(session=session, message="英伟达现在能不能买？"))
        payload = asyncio.run(service._build_payload(session=session, message="防守位呢？"))

        self.assertEqual(payload.answer_mode, "trade_plan_context")
        self.assertIn("防守位不是买入点", payload.answer)
        self.assertIn("风控预案", payload.answer)
        self.assertIn("未来", payload.answer)

    def test_trade_plan_llm_answer_rejects_extra_prices(self) -> None:
        trade_plan = make_trade_plan_payload()

        valid = StockAdvisorService._trade_plan_answer_is_consistent(
            (
                "英特尔当前价 52.00，低于第一买入区 69.88-74.99，观察区 46.49-51.03，"
                "防守位 38.45 是未来买入后的风控预案，不是买点。需要等待企稳确认，"
                "如果风险扩大则停止加仓。整体仍要先看风险和确认信号。"
            ),
            trade_plan,
        )
        invalid = StockAdvisorService._trade_plan_answer_is_consistent(
            "英特尔买点是 158.21 到 172.98，观察区 46.49 到 51.03。",
            trade_plan,
        )

        self.assertTrue(valid)
        self.assertFalse(invalid)

    def test_trade_plan_llm_answer_rejects_observation_contradiction(self) -> None:
        trade_plan = make_trade_plan_payload(current_price=48.0)

        invalid = StockAdvisorService._trade_plan_answer_is_consistent(
            "英特尔当前价 48.00。等待价格回到观察区 46.49-51.03 后再看。",
            trade_plan,
        )
        valid = StockAdvisorService._trade_plan_answer_is_consistent(
            (
                "英特尔当前价 48.00，已经在观察区 46.49-51.03 内，但还需要企稳确认。"
                "防守位 38.45 是未来买入后的风控预案，不是买点，风险扩大时再降低仓位。"
                "现在仍要等待确认，不能只看价格自动买入。"
            ),
            trade_plan,
        )

        self.assertFalse(invalid)
        self.assertTrue(valid)

    def test_trade_plan_llm_answer_rejects_defense_as_buy_point(self) -> None:
        trade_plan = make_trade_plan_payload()

        invalid = StockAdvisorService._trade_plan_answer_is_consistent(
            "英特尔防守位 38.45 是一个买入点，跌到这里可以考虑加仓。",
            trade_plan,
        )

        self.assertFalse(invalid)

    def test_trade_plan_llm_answer_rejects_wait_defense_as_current_action(self) -> None:
        trade_plan = make_trade_plan_payload()

        invalid = StockAdvisorService._trade_plan_answer_is_consistent(
            "英特尔防守位 38.45，有效跌破后停止加仓。",
            trade_plan,
        )
        valid = StockAdvisorService._trade_plan_answer_is_consistent(
            (
                "英特尔当前价 52.00，低于第一买入区 69.88-74.99。防守位 38.45 是未来买入后的"
                "风控预案，不是当前买点；如果未来已经按计划买入，有效跌破后停止加仓。"
                "执行前仍需要等待企稳确认，并持续检查风险是否扩大。"
            ),
            trade_plan,
        )

        self.assertFalse(invalid)
        self.assertTrue(valid)

    def test_switch_request_uses_llm_intent_for_new_stock(self) -> None:
        service = StockAdvisorService()
        service.intent_resolver = FakeIntentResolver(
            [
                ConversationIntent(
                    intent="trade_plan",
                    relation_to_last_trade="none",
                    target_ticker="MSFT",
                    target_name="微软",
                    confidence=0.95,
                ),
                ConversationIntent(
                    intent="trade_plan",
                    relation_to_last_trade="new_stock",
                    target_ticker="UNH",
                    target_name="联合健康",
                    confidence=0.95,
                ),
            ]
        )
        service.quote_lookup = fake_quote
        service.market_data = FakeMarketData()
        service.fundamentals = FakeFundamentals()
        session = service.sessions.create()

        first = asyncio.run(service._build_payload(session=session, message="微软现在买入节点是多少？"))
        second = asyncio.run(service._build_payload(session=session, message="换成联合健康呢？"))

        self.assertIsNotNone(first.trade_plan)
        self.assertIsNotNone(second.trade_plan)
        assert second.trade_plan is not None
        self.assertEqual(second.trade_plan.ticker, "UNH")
        self.assertIn(second.answer_mode, {"trade_plan_agent", "trade_plan_agent_ai"})

    def test_insufficient_trade_plan_uses_explicit_mode(self) -> None:
        service = StockAdvisorService()
        service.intent_resolver = FakeIntentResolver(
            [
                ConversationIntent(
                    intent="trade_plan",
                    relation_to_last_trade="none",
                    target_ticker="MSFT",
                    target_name="微软",
                    confidence=0.95,
                )
            ]
        )
        service.quote_lookup = fake_quote
        service.market_data = EmptyMarketData()
        service.fundamentals = FakeFundamentals()
        session = service.sessions.create()

        payload = asyncio.run(service._build_payload(session=session, message="微软现在买入节点是多少？"))

        self.assertEqual(payload.answer_mode, "trade_plan_insufficient")
        self.assertIsNone(payload.trade_plan)
        self.assertIn("数据不足", payload.answer)

    def test_stream_chat_emits_langgraph_progress_events(self) -> None:
        service = StockAdvisorService()
        service.intent_resolver = FakeIntentResolver(
            [
                ConversationIntent(
                    intent="trade_plan",
                    relation_to_last_trade="none",
                    target_ticker="NVDA",
                    target_name="英伟达",
                    confidence=0.95,
                )
            ]
        )
        service.quote_lookup = fake_quote
        service.market_data = FakeMarketData()
        service.fundamentals = FakeFundamentals()
        session = service.sessions.create()

        async def collect_events() -> list[str]:
            return [event async for event in service.stream_chat(session.session_id, "英伟达现在能不能买？")]

        events = asyncio.run(collect_events())
        joined = "".join(events)

        self.assertIn("event: meta", joined)
        self.assertIn("event: progress", joined)
        self.assertIn("RouterAgent", joined)
        self.assertIn("DataAgent", joined)
        self.assertIn("AnalysisAgent", joined)
        self.assertIn("WriterCriticAgent", joined)
        self.assertIn("run_tools", joined)
        self.assertIn("write_and_check", joined)
        self.assertIn("event: structured", joined)
        self.assertIn("event: done", joined)


class ConversationIntentResolverTest(unittest.TestCase):
    def test_context_followup_requires_previous_trade_context(self) -> None:
        resolver = FakeRawConversationIntentResolver(
            '{"intent":"context_followup","relation_to_last_trade":"followup","requested_field":"first_buy_zone","confidence":0.95}'
        )

        intent = asyncio.run(resolver.resolve("那第一买点是多少？", last_trade_context=None))

        self.assertEqual(intent.intent, "clarify")
        self.assertIn("上一只股票", intent.clarifying_question)

    def test_trade_plan_requires_valid_ticker(self) -> None:
        resolver = FakeRawConversationIntentResolver(
            '{"intent":"trade_plan","relation_to_last_trade":"none","target_ticker":"","target_name":"微软","confidence":0.95}'
        )

        intent = asyncio.run(resolver.resolve("微软现在买入节点是多少？", last_trade_context=None))

        self.assertEqual(intent.intent, "clarify")
        self.assertIn("股票", intent.clarifying_question)

    def test_accepts_llm_trade_plan_intent(self) -> None:
        resolver = FakeRawConversationIntentResolver(
            '{"intent":"trade_plan","relation_to_last_trade":"new_stock","target_ticker":"MSFT","target_name":"微软","requested_field":"full_plan","confidence":0.95}'
        )

        intent = asyncio.run(resolver.resolve("换成微软呢？", last_trade_context=None))

        self.assertEqual(intent.intent, "trade_plan")
        self.assertEqual(intent.target_ticker, "MSFT")
        self.assertEqual(intent.target_name, "微软")

    def test_accepts_tesla_trade_plan_intent(self) -> None:
        resolver = FakeRawConversationIntentResolver(
            '{"intent":"trade_plan","relation_to_last_trade":"none","target_ticker":"TSLA","target_name":"特斯拉","requested_field":"full_plan","confidence":0.95}'
        )

        intent = asyncio.run(resolver.resolve("现在买入特斯拉怎么样？", last_trade_context=None))

        self.assertEqual(intent.intent, "trade_plan")
        self.assertEqual(intent.target_ticker, "TSLA")
        self.assertEqual(intent.target_name, "特斯拉")


class FakeIntentResolver:
    def __init__(self, values: list[ConversationIntent]) -> None:
        self.values = values

    async def resolve(self, query: str, *, last_trade_context=None) -> ConversationIntent:
        if len(self.values) == 1:
            return self.values[0]
        return self.values.pop(0)


def make_trade_plan_payload(*, current_price: float = 52.0) -> TradePlanPayload:
    return TradePlanPayload(
        ticker="INTC",
        display_stock="英特尔",
        current_price=current_price,
        as_of="2026-04-24",
        action_state="wait",
        confidence="low",
        risk_state="normal",
        note="test",
        trend_stage="range_or_unclear",
        metrics=TradePlanMetrics(data_points=220),
        nodes=[
            TradePlanNode(
                key="observation",
                title="观察区",
                active=True,
                lower=46.49,
                upper=51.03,
                action="观察",
                formula="MA50 到 MA50 - 1 * ATR14",
            ),
            TradePlanNode(
                key="first_buy",
                title="第一买入区",
                active=True,
                lower=69.88,
                upper=74.99,
                action="分批试探",
                formula="60日高点回撤区间",
            ),
            TradePlanNode(
                key="deep_buy",
                title="深度买入区",
                active=True,
                lower=41.52,
                upper=41.52,
                action="深度回调参考",
                formula="120日高点回撤区间",
            ),
            TradePlanNode(
                key="defense",
                title="防守位",
                active=True,
                lower=38.45,
                upper=38.45,
                action="降低风险",
                formula="min(MA200, current price - ATR multiple * ATR14)",
            ),
        ],
        confirmation_condition="等待量能确认。",
        failure_condition="跌破防守位后停止加仓。",
        risk_adjustments=[],
    )


def make_chat_payload(session_id: str, *, answer: str) -> ChatPayload:
    return ChatPayload(
        answer=answer,
        scenarios=[],
        citations=[],
        disclaimer="test",
        session_state=SessionState(session_id=session_id, turn_count=1, summary="", model_mode="test"),
        answer_mode="test",
        trade_plan=None,
    )


class FakeRawConversationIntentResolver(ConversationIntentResolver):
    def __init__(self, raw: str) -> None:
        super().__init__(AppSettings(ollama_model="none"))
        self.raw = raw

    async def _call_model(self, *, query: str, last_trade_context=None) -> str:
        return self.raw


class FakeExplainer:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def explain(self, query: str) -> str:
        return self.answer


async def fake_quote(
    source_key: str,
    *,
    timeout: float,
    provider: str = "finnhub",
    api_key: str | None = None,
) -> QuoteSnapshot:
    return QuoteSnapshot(source_key=source_key, ticker=source_key.upper(), price=319, provider="finnhub")


class FakeMarketData:
    provider = "massive"

    async def get_daily_candles(self, source_key: str, *, lookback_days: int = 420) -> list[MarketCandle]:
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
            for index in range(220)
        ]


class EmptyMarketData:
    provider = "massive"

    async def get_daily_candles(self, source_key: str, *, lookback_days: int = 420) -> list[MarketCandle]:
        return []


class FakeFundamentals:
    async def get_snapshot(self, source_key: str) -> FundamentalSnapshot | None:
        return None


if __name__ == "__main__":
    unittest.main()
