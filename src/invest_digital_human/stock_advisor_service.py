from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .agent_planner import AgentPlan, TRADE_PLAN_TOOLS
from .chat_service import ChatSession, InvestmentChatService, SessionTurn
from .conversation_intent_resolver import (
    ConversationIntent,
    ConversationIntentResolver,
    LastTradeIntentContext,
)
from .llm_client import TRADE_PLAN_DISCLAIMER
from .llm_security_resolver import LLMSecurityResolver
from .schemas import ChatPayload, SessionState, TradePlanNode, TradePlanPayload
from .stock_advisor_explainer import CONCEPT_DISCLAIMER, StockAdvisorExplainer


ARTICLE_ONLY_DISCLAIMER = "以上内容仅用于功能路由提示，不构成投资建议。"


@dataclass(slots=True)
class LastTradeContext:
    ticker: str
    display_stock: str
    payload: ChatPayload
    updated_at: datetime


class StockAdvisorService(InvestmentChatService):
    """Stock-advisor-only service with an LLM conversation judgment node."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.intent_resolver = ConversationIntentResolver(self.settings)
        self.security_resolver = LLMSecurityResolver(self.settings)
        self.explainer = StockAdvisorExplainer(self.settings)
        self._last_trade_context_by_session: dict[str, LastTradeContext] = {}

    async def _build_payload(self, *, session: ChatSession, message: str) -> ChatPayload:
        self._compress_session(session)
        context = self._last_trade_context_by_session.get(session.session_id)
        intent = await self._resolve_conversation_intent(message=message, context=context)

        if intent.intent == "context_followup":
            return self._build_context_followup_payload(session=session, message=message, intent=intent)

        if intent.intent == "trade_plan":
            payload = await self._build_resolved_trade_plan(
                session=session,
                message=message,
                ticker=intent.target_ticker,
                display_name=intent.target_name or intent.target_ticker,
            )
            self._remember_trade_context(session, payload)
            return payload

        if intent.intent == "article_history":
            return self._build_simple_payload(
                session=session,
                message=message,
                answer=(
                    "股票建议项目不查询历史文章。"
                    "如果你想查作者文章里的历史观点或历史节点，请启动历史文章项目。"
                ),
                mode="stock_advisor_llm_router",
                disclaimer=ARTICLE_ONLY_DISCLAIMER,
            )

        if intent.intent == "concept_explain":
            return await self._build_concept_payload(session=session, message=message)

        answer = intent.clarifying_question or "请补充股票名称或代码，或者说明你想了解的概念。"
        return self._build_simple_payload(
            session=session,
            message=message,
            answer=answer,
            mode="stock_advisor_clarify",
            disclaimer=CONCEPT_DISCLAIMER,
        )

    async def _resolve_conversation_intent(
        self,
        *,
        message: str,
        context: LastTradeContext | None,
    ) -> ConversationIntent:
        try:
            return await self.intent_resolver.resolve(
                message,
                last_trade_context=_to_intent_context(context),
            )
        except Exception:
            return ConversationIntent(
                intent="clarify",
                relation_to_last_trade="ambiguous",
                confidence=0.0,
                clarifying_question="我暂时无法稳定识别你的问题，请补充股票名称或具体想问的内容。",
                reason="intent resolver failed",
            )

    async def _build_resolved_trade_plan(
        self,
        *,
        session: ChatSession,
        message: str,
        ticker: str,
        display_name: str,
    ) -> ChatPayload:
        agent_plan = AgentPlan(
            intent="trade_plan",
            source_key=ticker.lower(),
            display_stock=display_name or ticker,
            tools=TRADE_PLAN_TOOLS,
            use_articles=False,
            raw_query=message,
        )
        return await self._build_trade_plan_payload(session=session, agent_plan=agent_plan)

    def _remember_trade_context(self, session: ChatSession, payload: ChatPayload) -> None:
        if payload.trade_plan is None:
            return
        self._last_trade_context_by_session[session.session_id] = LastTradeContext(
            ticker=payload.trade_plan.ticker,
            display_stock=payload.trade_plan.display_stock,
            payload=payload.model_copy(deep=True),
            updated_at=datetime.now(timezone.utc),
        )

    def _build_context_followup_payload(
        self,
        *,
        session: ChatSession,
        message: str,
        intent: ConversationIntent,
    ) -> ChatPayload:
        context = self._last_trade_context_by_session.get(session.session_id)
        if context is None or context.payload.trade_plan is None:
            return self._build_simple_payload(
                session=session,
                message=message,
                answer="我还没有上一只股票的交易计划，请先告诉我要分析哪只股票。",
                mode="stock_advisor_clarify",
                disclaimer=CONCEPT_DISCLAIMER,
            )

        trade_plan = context.payload.trade_plan
        answer = _render_context_followup_answer(
            requested_field=intent.requested_field or "full_plan",
            trade_plan=trade_plan,
        )
        session.turns.append(SessionTurn(role="user", content=message))
        session.turns.append(SessionTurn(role="assistant", content=answer))
        self._compress_session(session)
        return ChatPayload(
            answer=answer,
            scenarios=[],
            citations=[],
            disclaimer=TRADE_PLAN_DISCLAIMER,
            session_state=SessionState(
                session_id=session.session_id,
                turn_count=session.turn_count,
                summary=session.summary,
                model_mode="stock_advisor_context",
            ),
            answer_mode="trade_plan_context",
            trade_plan=None,
        )

    async def _build_concept_payload(self, *, session: ChatSession, message: str) -> ChatPayload:
        try:
            answer = await self.explainer.explain(message)
            mode = "stock_advisor_explainer"
        except Exception:
            answer = (
                "我暂时不能调用解释模型。"
                "如果你想计算某只股票的买入节点，请直接输入股票名称或代码。"
            )
            mode = "stock_advisor_llm_router"
        return self._build_simple_payload(
            session=session,
            message=message,
            answer=answer,
            mode=mode,
            disclaimer=CONCEPT_DISCLAIMER,
        )

    def _build_simple_payload(
        self,
        *,
        session: ChatSession,
        message: str,
        answer: str,
        mode: str,
        disclaimer: str,
    ) -> ChatPayload:
        session.turns.append(SessionTurn(role="user", content=message))
        session.turns.append(SessionTurn(role="assistant", content=answer))
        self._compress_session(session)
        return ChatPayload(
            answer=answer,
            scenarios=[],
            citations=[],
            disclaimer=disclaimer,
            session_state=SessionState(
                session_id=session.session_id,
                turn_count=session.turn_count,
                summary=session.summary,
                model_mode=mode,
            ),
            answer_mode=mode,
            trade_plan=None,
        )


def _to_intent_context(context: LastTradeContext | None) -> LastTradeIntentContext | None:
    if context is None or context.payload.trade_plan is None:
        return None
    trade_plan = context.payload.trade_plan
    return LastTradeIntentContext(
        ticker=trade_plan.ticker,
        display_stock=trade_plan.display_stock,
        current_price=trade_plan.current_price,
        action_state=trade_plan.action_state,
        confidence=trade_plan.confidence,
        nodes=[
            {
                "key": node.key,
                "title": node.title,
                "active": node.active,
                "lower": node.lower,
                "upper": node.upper,
                "action": node.action,
            }
            for node in trade_plan.nodes
        ],
        confirmation_condition=trade_plan.confirmation_condition,
        failure_condition=trade_plan.failure_condition,
    )


def _render_context_followup_answer(*, requested_field: str, trade_plan: TradePlanPayload) -> str:
    if requested_field == "watch_zone":
        return _render_node_answer(trade_plan, "observation")
    if requested_field == "first_buy_zone":
        return _render_node_answer(trade_plan, "first_buy")
    if requested_field == "deep_buy_zone":
        return _render_node_answer(trade_plan, "deep_buy")
    if requested_field == "defense":
        return _render_node_answer(trade_plan, "defense")
    if requested_field == "confirmation":
        return f"{trade_plan.display_stock} 的确认条件是：{trade_plan.confirmation_condition}"
    if requested_field == "invalidation":
        return f"{trade_plan.display_stock} 的失效条件是：{trade_plan.failure_condition}"
    if requested_field == "rationale":
        return _render_rationale_answer(trade_plan)
    return _render_full_plan_summary(trade_plan)


def _render_node_answer(trade_plan: TradePlanPayload, key: str) -> str:
    stock = trade_plan.display_stock.strip()
    node = _node_by_key(trade_plan, key)
    if node is None:
        return f"{stock} 当前没有 {key} 对应的节点数据。"
    if not node.active:
        return (
            f"{stock}的{node.title}当前未激活。"
            f"系统结论是“{trade_plan.action_state}”，需要先满足确认条件：{trade_plan.confirmation_condition}"
        )
    if key == "first_buy":
        return (
            f"{stock}的第一买入区是 {_format_node_value(node)}。"
            "这不是“立刻买”的指令，而是价格回到这个区域后，才考虑分批试探的位置。"
            f"还要看确认条件：{trade_plan.confirmation_condition}"
        )
    if key == "defense":
        return (
            f"{stock}的防守位是 {_format_node_value(node)}。"
            "防守位不是买入点，也不是预测底部。"
            "它是未来按计划买入后才使用的风控预案/交易计划失效线。"
            "如果你未来已经买入或已有仓位，价格有效跌破这里，就应该停止加仓、降低风险或重新评估。"
        )
    return (
        f"{stock}的{node.title}是 {_format_node_value(node)}。"
        f"动作：{node.action}。"
        f"确认条件：{trade_plan.confirmation_condition}"
    )


def _render_rationale_answer(trade_plan: TradePlanPayload) -> str:
    stock = trade_plan.display_stock.strip()
    first_buy = _node_by_key(trade_plan, "first_buy")
    first_buy_text = _format_node_value(first_buy) if first_buy and first_buy.active else "未激活"
    reasons = "；".join(_rationale_reasons(trade_plan))
    return (
        f"不直接现在买，主要因为{stock}还没有满足系统的买入确认条件。"
        f"当前行动状态是“{_action_state_label(trade_plan.action_state)}”，计划等级是“{_plan_level_label(trade_plan.confidence)}”。"
        f"第一买入区是 {first_buy_text}，没到或没有企稳前不适合追高。"
        f"主要原因：{reasons}。"
        f"后续要等这个确认条件：{trade_plan.confirmation_condition}"
    )


def _rationale_reasons(trade_plan: TradePlanPayload) -> list[str]:
    reasons: list[str] = []
    first_buy = _node_by_key(trade_plan, "first_buy")
    if first_buy and first_buy.active and first_buy.upper is not None:
        if trade_plan.current_price > first_buy.upper:
            reasons.append("当前价还在第一买入区上方，追高的性价比不够")
        elif first_buy.lower <= trade_plan.current_price <= first_buy.upper:
            reasons.append("当前价虽然接近买入区，但仍需要企稳确认")
    elif first_buy and not first_buy.active:
        reasons.append("第一买入区当前未激活，系统不强行给进攻买点")

    raw_reasons = [_clean_reason(reason) for reason in trade_plan.risk_adjustments if reason]
    preferred_keywords = (
        "RSI",
        "过热",
        "MA200",
        "弱势",
        "财报",
        "事件风险",
        "大盘",
        "板块",
        "放量下跌",
    )
    for keyword in preferred_keywords:
        for reason in raw_reasons:
            if _is_positive_or_non_risk_reason(reason):
                continue
            if keyword in reason and reason not in reasons:
                reasons.append(reason)
                break
        if len(reasons) >= 3:
            break

    if len(reasons) < 3:
        for reason in raw_reasons:
            if _is_analyst_rating_reason(reason):
                continue
            if _is_positive_or_non_risk_reason(reason):
                continue
            if reason not in reasons:
                reasons.append(reason)
            if len(reasons) >= 3:
                break

    return reasons[:3] or ["当前条件还不充分"]


def _is_analyst_rating_reason(reason: str) -> bool:
    return "分析师" in reason or "评级" in reason


def _is_positive_or_non_risk_reason(reason: str) -> bool:
    return any(fragment in reason for fragment in ("加分", "正向", "未触发", "占优", "校准置信度"))


def _render_full_plan_summary(trade_plan: TradePlanPayload) -> str:
    stock = trade_plan.display_stock.strip()
    nodes = []
    for key in ("observation", "first_buy", "deep_buy", "defense"):
        node = _node_by_key(trade_plan, key)
        if node is None:
            continue
        value = _format_node_value(node) if node.active else "未激活"
        nodes.append(f"{node.title}：{value}")
    node_text = "；".join(nodes) if nodes else "暂无完整节点"
    return (
        f"继续按上一轮 {stock} 的交易计划看："
        f"当前价 {trade_plan.current_price:.2f}，行动状态“{_action_state_label(trade_plan.action_state)}”，"
        f"计划等级“{_plan_level_label(trade_plan.confidence)}”。{node_text}。"
        f"确认条件：{trade_plan.confirmation_condition}"
    )


def _node_by_key(trade_plan: TradePlanPayload, key: str) -> TradePlanNode | None:
    for node in trade_plan.nodes:
        if node.key == key:
            return node
    return None


def _format_node_value(node: TradePlanNode) -> str:
    if node.lower is None:
        return "未激活"
    if node.upper is None or node.lower == node.upper:
        return f"{node.lower:.2f}"
    return f"{node.lower:.2f}-{node.upper:.2f}"


def _action_state_label(value: str) -> str:
    return {
        "wait": "等待",
        "starter_allowed": "允许小仓位试探",
        "add_allowed": "允许加仓",
        "risk_reduce": "降低风险",
    }.get(value, value)


def _plan_level_label(value: str) -> str:
    return {"low": "观察级", "medium": "计划级", "high": "执行级"}.get(value, value)


def _clean_reason(reason: str) -> str:
    return reason.strip().rstrip("。；;")
