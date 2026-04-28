from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agent_planner import AgentPlan, TRADE_PLAN_TOOLS
from .agent_tools import TradePlanToolBundle, TradePlanToolRunner
from .chat_service import ChatSession, SessionTurn, _has_minimum_trade_plan_content
from .conversation_intent_resolver import (
    ConversationIntent,
    ConversationIntentResolver,
    LastTradeIntentContext,
)
from .llm_client import TRADE_PLAN_DISCLAIMER
from .schemas import ChatPayload, SessionState, TradePlanNode, TradePlanPayload
from .stock_advisor_explainer import CONCEPT_DISCLAIMER, StockAdvisorExplainer
from .trade_plan_agent import TradePlanResult, build_trade_plan


ARTICLE_ONLY_DISCLAIMER = "以上内容仅用于功能路由提示，不构成投资建议。"

PROGRESS_EVENTS = {
    "router_agent": {
        "agent": "RouterAgent",
        "step": "route",
        "label": "Router Agent 正在理解问题",
    },
    "context_agent": {
        "agent": "RouterAgent",
        "step": "context_followup",
        "label": "Router Agent 正在读取上一轮交易计划",
    },
    "concept_agent": {
        "agent": "RouterAgent",
        "step": "concept_explain",
        "label": "Router Agent 正在整理概念解释",
    },
    "article_history_agent": {
        "agent": "RouterAgent",
        "step": "article_history",
        "label": "Router Agent 正在处理历史文章请求",
    },
    "clarify_agent": {
        "agent": "RouterAgent",
        "step": "clarify",
        "label": "Router Agent 正在生成澄清问题",
    },
    "data_agent": {
        "agent": "DataAgent",
        "step": "run_tools",
        "label": "Data Agent 正在获取行情和K线",
    },
    "analysis_agent": {
        "agent": "AnalysisAgent",
        "step": "build_plan",
        "label": "Analysis Agent 正在计算交易计划",
    },
    "writer_critic_agent": {
        "agent": "WriterCriticAgent",
        "step": "write_and_check",
        "label": "Writer/Critic Agent 正在生成并校验回答",
    },
    "payload_agent": {
        "agent": "WriterCriticAgent",
        "step": "build_payload",
        "label": "Writer/Critic Agent 正在整理最终结果",
    },
}


@dataclass(slots=True)
class LastTradeContext:
    ticker: str
    display_stock: str
    payload: ChatPayload
    updated_at: datetime


class AgentState(TypedDict, total=False):
    session: ChatSession
    message: str
    last_trade_context: LastTradeContext | None
    intent: ConversationIntent
    agent_plan: AgentPlan
    tools: TradePlanToolBundle
    trade_result: TradePlanResult
    answer: str
    scenarios: list[Any]
    disclaimer: str
    model_mode: str
    answer_mode: str
    payload: ChatPayload
    agent_trace: list[dict[str, object]]
    error: str


class StockAdvisorGraph:
    """Controlled multi-agent stock advisor graph.

    LLM calls are limited to RouterAgent and WriterCriticAgent. Data retrieval,
    price nodes, scoring, and structured trade plans remain deterministic.
    """

    def __init__(
        self,
        *,
        service: Any,
        intent_resolver: ConversationIntentResolver,
        explainer: StockAdvisorExplainer,
    ) -> None:
        self.service = service
        self.intent_resolver = intent_resolver
        self.explainer = explainer
        self.graph = self._compile_graph()

    async def stream(
        self,
        *,
        session: ChatSession,
        message: str,
        last_trade_context: LastTradeContext | None,
    ) -> AsyncIterator[tuple[str, dict[str, object] | ChatPayload]]:
        state: AgentState = {
            "session": session,
            "message": message,
            "last_trade_context": last_trade_context,
            "agent_trace": [],
        }
        final_payload: ChatPayload | None = None
        async for update in self.graph.astream(state, stream_mode="updates"):
            for node_name, node_update in update.items():
                progress = PROGRESS_EVENTS.get(node_name)
                if progress is not None:
                    yield ("progress", dict(progress))
                if isinstance(node_update, dict) and isinstance(node_update.get("payload"), ChatPayload):
                    final_payload = node_update["payload"]
        if final_payload is None:
            raise RuntimeError("StockAdvisorGraph finished without a ChatPayload")
        yield ("payload", final_payload)

    async def invoke(
        self,
        *,
        session: ChatSession,
        message: str,
        last_trade_context: LastTradeContext | None,
    ) -> ChatPayload:
        state: AgentState = {
            "session": session,
            "message": message,
            "last_trade_context": last_trade_context,
            "agent_trace": [],
        }
        result = await self.graph.ainvoke(state)
        payload = result.get("payload")
        if not isinstance(payload, ChatPayload):
            raise RuntimeError("StockAdvisorGraph finished without a ChatPayload")
        return payload

    def _compile_graph(self) -> CompiledStateGraph:
        graph: StateGraph = StateGraph(AgentState)
        graph.add_node("router_agent", self._router_agent)
        graph.add_node("context_agent", self._context_agent)
        graph.add_node("concept_agent", self._concept_agent)
        graph.add_node("article_history_agent", self._article_history_agent)
        graph.add_node("clarify_agent", self._clarify_agent)
        graph.add_node("data_agent", self._data_agent)
        graph.add_node("analysis_agent", self._analysis_agent)
        graph.add_node("writer_critic_agent", self._writer_critic_agent)
        graph.add_node("payload_agent", self._payload_agent)

        graph.set_entry_point("router_agent")
        graph.add_conditional_edges(
            "router_agent",
            self._route_from_intent,
            {
                "context_followup": "context_agent",
                "concept_explain": "concept_agent",
                "article_history": "article_history_agent",
                "clarify": "clarify_agent",
                "trade_plan": "data_agent",
            },
        )
        graph.add_conditional_edges(
            "data_agent",
            self._route_after_data,
            {
                "cached": END,
                "analysis": "analysis_agent",
            },
        )
        graph.add_conditional_edges(
            "analysis_agent",
            self._route_after_analysis,
            {
                "insufficient": END,
                "write": "writer_critic_agent",
            },
        )
        graph.add_edge("writer_critic_agent", "payload_agent")
        graph.add_edge("payload_agent", END)
        graph.add_edge("context_agent", END)
        graph.add_edge("concept_agent", END)
        graph.add_edge("article_history_agent", END)
        graph.add_edge("clarify_agent", END)
        return graph.compile()

    async def _router_agent(self, state: AgentState) -> dict[str, object]:
        try:
            intent = await self.intent_resolver.resolve(
                state["message"],
                last_trade_context=_to_intent_context(state.get("last_trade_context")),
            )
            summary = f"intent={intent.intent}, ticker={intent.target_ticker or '-'}"
        except Exception:
            intent = ConversationIntent(
                intent="clarify",
                relation_to_last_trade="ambiguous",
                confidence=0.0,
                clarifying_question="我暂时无法稳定识别你的问题，请补充股票名称或具体想问的内容。",
                reason="intent resolver failed",
            )
            summary = "intent resolver failed; routed to clarify"
        return {
            "intent": intent,
            "agent_trace": _append_trace(state, "RouterAgent", "ok", summary),
        }

    def _route_from_intent(self, state: AgentState) -> str:
        return state["intent"].intent

    def _route_after_data(self, state: AgentState) -> str:
        return "cached" if isinstance(state.get("payload"), ChatPayload) else "analysis"

    def _route_after_analysis(self, state: AgentState) -> str:
        result = state.get("trade_result")
        if result is None or result.trade_plan is None:
            return "insufficient"
        return "write"

    def _context_agent(self, state: AgentState) -> dict[str, object]:
        session = state["session"]
        message = state["message"]
        context = state.get("last_trade_context")
        intent = state["intent"]
        if context is None or context.payload.trade_plan is None:
            answer = "我还没有上一只股票的交易计划，请先告诉我要分析哪只股票。"
            payload = _simple_payload(
                service=self.service,
                session=session,
                message=message,
                answer=answer,
                mode="stock_advisor_clarify",
                disclaimer=CONCEPT_DISCLAIMER,
            )
        else:
            answer = render_context_followup_answer(
                requested_field=intent.requested_field or "full_plan",
                trade_plan=context.payload.trade_plan,
            )
            payload = _simple_payload(
                service=self.service,
                session=session,
                message=message,
                answer=answer,
                mode="stock_advisor_context",
                disclaimer=TRADE_PLAN_DISCLAIMER,
                answer_mode="trade_plan_context",
            )
        return {
            "payload": payload,
            "agent_trace": _append_trace(state, "RouterAgent", "ok", "answered context follow-up"),
        }

    async def _concept_agent(self, state: AgentState) -> dict[str, object]:
        try:
            answer = await self.explainer.explain(state["message"])
            mode = "stock_advisor_explainer"
            summary = "concept explained by LLM"
        except Exception:
            answer = "我暂时不能调用解释模型。如果你想计算某只股票的买入节点，请直接输入股票名称或代码。"
            mode = "stock_advisor_llm_router"
            summary = "concept explainer failed; used fallback"
        payload = _simple_payload(
            service=self.service,
            session=state["session"],
            message=state["message"],
            answer=answer,
            mode=mode,
            disclaimer=CONCEPT_DISCLAIMER,
        )
        return {
            "payload": payload,
            "agent_trace": _append_trace(state, "RouterAgent", "ok", summary),
        }

    def _article_history_agent(self, state: AgentState) -> dict[str, object]:
        payload = _simple_payload(
            service=self.service,
            session=state["session"],
            message=state["message"],
            answer="股票建议项目不查询历史文章。如果你想查作者文章里的历史观点或历史节点，请启动历史文章项目。",
            mode="stock_advisor_llm_router",
            disclaimer=ARTICLE_ONLY_DISCLAIMER,
        )
        return {
            "payload": payload,
            "agent_trace": _append_trace(state, "RouterAgent", "ok", "article history request isolated"),
        }

    def _clarify_agent(self, state: AgentState) -> dict[str, object]:
        intent = state["intent"]
        payload = _simple_payload(
            service=self.service,
            session=state["session"],
            message=state["message"],
            answer=intent.clarifying_question or "请补充股票名称或代码，或者说明你想了解的概念。",
            mode="stock_advisor_clarify",
            disclaimer=CONCEPT_DISCLAIMER,
        )
        return {
            "payload": payload,
            "agent_trace": _append_trace(state, "RouterAgent", "ok", "asked for clarification"),
        }

    async def _data_agent(self, state: AgentState) -> dict[str, object]:
        intent = state["intent"]
        source_key = intent.target_ticker.lower()
        agent_plan = AgentPlan(
            intent="trade_plan",
            source_key=source_key,
            display_stock=intent.target_name or intent.target_ticker,
            tools=TRADE_PLAN_TOOLS,
            use_articles=False,
            raw_query=state["message"],
        )
        cached = self.service._get_cached_trade_plan(source_key)
        if cached is not None:
            payload = self.service._record_cached_payload(
                session=state["session"],
                query=state["message"],
                payload=cached,
            )
            return {
                "agent_plan": agent_plan,
                "payload": payload,
                "agent_trace": _append_trace(state, "DataAgent", "cached", f"{source_key} cache hit"),
            }
        tools = await TradePlanToolRunner(
            market_data=self.service.market_data,
            fundamentals=self.service.fundamentals,
            quote_timeout=self.service.settings.quote_lookup_timeout,
            finnhub_api_key=self.service.settings.finnhub_api_key,
            quote_lookup_fn=self.service.quote_lookup,
        ).run(source_key)
        tool_status = tools.as_log_facts()
        return {
            "agent_plan": agent_plan,
            "tools": tools,
            "agent_trace": _append_trace(
                state,
                "DataAgent",
                "ok",
                f"quote={tools.quote.ok}, candles={tools.candles.ok}, fundamentals={tools.fundamentals.ok}",
                tool_status=tool_status,
            ),
        }

    def _analysis_agent(self, state: AgentState) -> dict[str, object]:
        session = state["session"]
        agent_plan = state["agent_plan"]
        tools = state["tools"]
        result = build_trade_plan(
            display_stock=agent_plan.display_stock or agent_plan.source_key.upper(),
            quote=tools.quote.data,
            candles=tools.candles.data or [],
            fundamentals=tools.fundamentals.data,
            market_context=tools.market_context.data,
            calibration=self.service.calibrations.get(agent_plan.source_key or ""),
        )
        if result.trade_plan is None:
            session.turns.append(SessionTurn(role="user", content=agent_plan.raw_query))
            session.turns.append(SessionTurn(role="assistant", content=result.answer))
            self.service._compress_session(session)
            payload = ChatPayload(
                answer=result.answer,
                scenarios=[],
                citations=[],
                disclaimer=TRADE_PLAN_DISCLAIMER,
                session_state=SessionState(
                    session_id=session.session_id,
                    turn_count=session.turn_count,
                    summary=session.summary,
                    model_mode="trade_plan_insufficient",
                ),
                answer_mode="trade_plan_insufficient",
                trade_plan=None,
            )
            return {
                "trade_result": result,
                "payload": payload,
                "agent_trace": _append_trace(state, "AnalysisAgent", "insufficient", "trade plan data insufficient"),
            }
        summary = f"action={result.action_state}, confidence={result.confidence}"
        return {
            "trade_result": result,
            "answer": result.answer,
            "scenarios": result.scenarios,
            "disclaimer": TRADE_PLAN_DISCLAIMER,
            "model_mode": "trade_plan_agent",
            "answer_mode": "trade_plan_agent",
            "agent_trace": _append_trace(state, "AnalysisAgent", "ok", summary),
        }

    async def _writer_critic_agent(self, state: AgentState) -> dict[str, object]:
        result = state["trade_result"]
        update: dict[str, object] = {
            "answer": result.answer,
            "scenarios": result.scenarios,
            "disclaimer": TRADE_PLAN_DISCLAIMER,
            "model_mode": "trade_plan_agent",
            "answer_mode": "trade_plan_agent",
        }
        summary = "used deterministic answer"
        if (
            self.service.settings.enable_trade_plan_llm
            and self.service.llm_client.mode != "fallback"
            and result.trade_plan is not None
        ):
            try:
                generation = await self.service.llm_client.generate_trade_plan_answer(
                    query=state["agent_plan"].raw_query,
                    deterministic_answer=result.answer,
                    trade_facts=result.facts,
                    scenarios=result.scenarios,
                    session_context=self.service._render_context(state["session"]),
                )
                answer = generation.answer
                if not self.service._trade_plan_answer_is_consistent(answer, result.trade_plan):
                    raise ValueError("LLM trade plan answer failed deterministic validation")
                if not _has_minimum_trade_plan_content(answer, result.trade_plan):
                    raise ValueError("LLM trade plan answer is too thin")
                update = {
                    "answer": self.service._ensure_trade_plan_sections(answer, result.trade_plan),
                    "scenarios": generation.scenarios,
                    "disclaimer": generation.disclaimer,
                    "model_mode": generation.mode,
                    "answer_mode": "trade_plan_agent_ai",
                }
                summary = "LLM answer accepted by critic"
            except Exception:
                summary = "LLM answer rejected; used deterministic answer"
        update["agent_trace"] = _append_trace(state, "WriterCriticAgent", "ok", summary)
        return update

    def _payload_agent(self, state: AgentState) -> dict[str, object]:
        session = state["session"]
        result = state["trade_result"]
        answer = str(state.get("answer") or result.answer)
        session.turns.append(SessionTurn(role="user", content=state["agent_plan"].raw_query))
        session.turns.append(SessionTurn(role="assistant", content=answer))
        self.service._compress_session(session)
        payload = ChatPayload(
            answer=answer,
            scenarios=list(state.get("scenarios") or result.scenarios),
            citations=[],
            disclaimer=str(state.get("disclaimer") or TRADE_PLAN_DISCLAIMER),
            session_state=SessionState(
                session_id=session.session_id,
                turn_count=session.turn_count,
                summary=session.summary,
                model_mode=str(state.get("model_mode") or "trade_plan_agent"),
            ),
            answer_mode=str(state.get("answer_mode") or "trade_plan_agent"),
            trade_plan=result.trade_plan,
        )
        self.service._set_cached_trade_plan(state["agent_plan"].source_key or "", payload)
        return {
            "payload": payload,
            "agent_trace": _append_trace(state, "WriterCriticAgent", "ok", "final payload built"),
        }


def _append_trace(
    state: AgentState,
    name: str,
    status: str,
    summary: str,
    *,
    tool_status: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    item: dict[str, object] = {
        "name": name,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if tool_status is not None:
        item["tool_status"] = tool_status
    return [*(state.get("agent_trace") or []), item]


def _simple_payload(
    *,
    service: Any,
    session: ChatSession,
    message: str,
    answer: str,
    mode: str,
    disclaimer: str,
    answer_mode: str | None = None,
) -> ChatPayload:
    session.turns.append(SessionTurn(role="user", content=message))
    session.turns.append(SessionTurn(role="assistant", content=answer))
    service._compress_session(session)
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
        answer_mode=answer_mode or mode,
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


def remember_trade_context(payload: ChatPayload) -> LastTradeContext | None:
    if payload.trade_plan is None:
        return None
    return LastTradeContext(
        ticker=payload.trade_plan.ticker,
        display_stock=payload.trade_plan.display_stock,
        payload=payload.model_copy(deep=True),
        updated_at=datetime.now(timezone.utc),
    )


def render_context_followup_answer(*, requested_field: str, trade_plan: TradePlanPayload) -> str:
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
            f"{stock} 的{node.title}当前暂不参考。"
            f"系统结论是“{_action_state_label(trade_plan.action_state)}”，"
            f"需要先满足确认条件：{trade_plan.confirmation_condition}"
        )
    if key == "first_buy":
        return (
            f"{stock} 的第一买入区是 {_format_node_value(node)}。"
            "这不是“立刻买”的指令，而是价格回到这个区间后，才考虑分批试探的位置。"
            f"还要看确认条件：{trade_plan.confirmation_condition}"
        )
    if key == "defense":
        return (
            f"{stock} 的防守位是 {_format_node_value(node)}。"
            "防守位不是买入点，也不是预测底部。"
            "它是未来按计划买入后才使用的风控预案/交易计划失效线。"
            "如果未来已经买入或已有仓位，价格有效跌破这里，就应该停止加仓、降低风险或重新评估。"
        )
    return (
        f"{stock} 的{node.title}是 {_format_node_value(node)}。"
        f"动作：{node.action}。"
        f"确认条件：{trade_plan.confirmation_condition}"
    )


def _render_rationale_answer(trade_plan: TradePlanPayload) -> str:
    stock = trade_plan.display_stock.strip()
    first_buy = _node_by_key(trade_plan, "first_buy")
    first_buy_text = _format_node_value(first_buy) if first_buy and first_buy.active else "暂不参考"
    reasons = "；".join(_rationale_reasons(trade_plan))
    return (
        f"不直接现在买，主要因为 {stock} 还没有满足系统的买入确认条件。"
        f"当前行动状态是“{_action_state_label(trade_plan.action_state)}”，"
        f"计划等级是“{_plan_level_label(trade_plan.confidence)}”。"
        f"第一买入区是 {first_buy_text}，没到或没有企稳前不适合追高。"
        f"主要原因：{reasons}。"
        f"后续要等这个确认条件：{trade_plan.confirmation_condition}"
    )


def _rationale_reasons(trade_plan: TradePlanPayload) -> list[str]:
    reasons: list[str] = []
    first_buy = _node_by_key(trade_plan, "first_buy")
    if first_buy and first_buy.active and first_buy.upper is not None:
        if trade_plan.current_price > first_buy.upper:
            reasons.append("当前价还在第一买入区上方，追高的性价比不高")
        elif first_buy.lower and first_buy.lower <= trade_plan.current_price <= first_buy.upper:
            reasons.append("当前价虽然接近买入区，但仍需要企稳确认")
    elif first_buy and not first_buy.active:
        reasons.append("第一买入区当前未激活，系统不强行给进攻买点")
    for reason in trade_plan.risk_adjustments:
        cleaned = reason.strip().rstrip("。；;")
        if cleaned and cleaned not in reasons:
            reasons.append(cleaned)
        if len(reasons) >= 3:
            break
    return reasons[:3] or ["当前条件还不充分"]


def _render_full_plan_summary(trade_plan: TradePlanPayload) -> str:
    stock = trade_plan.display_stock.strip()
    nodes = []
    for key in ("observation", "first_buy", "deep_buy", "defense"):
        node = _node_by_key(trade_plan, key)
        if node is None:
            continue
        value = _format_node_value(node) if node.active else "暂不参考"
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


def _format_node_value(node: TradePlanNode | None) -> str:
    if node is None or node.lower is None:
        return "暂不参考"
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
