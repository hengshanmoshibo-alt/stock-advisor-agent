from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from uuid import uuid4

from .agent_planner import AgentPlan, AgentPlanner
from .agent_tools import TradePlanToolRunner
from .calibration import TradeNodeCalibrationStore
from .config import AppSettings
from .embeddings import EmbeddingSettings, build_dense_embedder
from .fundamental_data import FundamentalDataClient
from .llm_client import (
    DEFAULT_DISCLAIMER,
    TRADE_PLAN_DISCLAIMER,
    build_citations_from_hits,
    build_llm_client,
)
from .market_data import MarketDataClient
from .market_context import build_market_context
from .quote_lookup import build_quote_distance_summary, lookup_quote, quote_lookup_requested
from .reranking import RerankerSettings, build_reranker
from .retrieval import RAGIndex, SearchHit
from .schemas import ChatPayload, HealthResponse, SessionState
from .stock_node_answering import build_stock_node_answer
from .stock_nodes import StockNodeKnowledgeBase, StockNodeQuery, StockNodeRecord
from .technical_nodes import (
    calculated_nodes_requested,
    calculate_technical_node_plan,
    calculated_node_extra_facts,
    format_enhanced_calculated_node_answer,
)
from .trade_plan_agent import build_trade_plan


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionTurn:
    role: str
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


@dataclass(slots=True)
class ChatSession:
    session_id: str
    turns: list[SessionTurn] = field(default_factory=list)
    summary: str = ""
    title: str = "新对话"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    messages_for_ui: list[dict[str, object]] = field(default_factory=list)

    @property
    def turn_count(self) -> int:
        return sum(1 for turn in self.turns if turn.role == "user")


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(self) -> ChatSession:
        session = ChatSession(session_id=uuid4().hex)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        return self._sessions[session_id]

    def list_recent(self) -> list[ChatSession]:
        return sorted(self._sessions.values(), key=lambda session: session.updated_at, reverse=True)


class InvestmentChatService:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings.from_env()
        self.embedder = build_dense_embedder(
            EmbeddingSettings(
                backend=self.settings.embedding_backend,
                model_name=self.settings.embedding_model,
                api_key=self.settings.embedding_api_key,
                base_url=self.settings.embedding_base_url,
                timeout=self.settings.request_timeout,
                hf_home=self.settings.huggingface_home,
            )
        )
        if self.settings.index_path.exists():
            self.index = RAGIndex.load(self.settings.index_path)
        else:
            LOGGER.warning("rag_index_missing_using_empty_index", extra={"index_path": str(self.settings.index_path)})
            self.index = RAGIndex(chunks=[], doc_freq={}, avg_doc_len=1.0)
        self.stock_nodes = (
            StockNodeKnowledgeBase(self.settings.stock_node_path)
            if self.settings.enable_stock_node_rag
            else None
        )
        self.agent_planner = AgentPlanner(self.stock_nodes)
        self.quote_lookup = lookup_quote
        market_data_api_key = (
            self.settings.massive_api_key
            if self.settings.market_data_provider.strip().lower() == "massive"
            else self.settings.finnhub_api_key
        )
        self.market_data = MarketDataClient(
            provider=self.settings.market_data_provider,
            api_key=market_data_api_key,
            timeout=self.settings.quote_lookup_timeout,
        )
        self.fundamentals = FundamentalDataClient(
            api_key=self.settings.finnhub_api_key,
            timeout=self.settings.quote_lookup_timeout,
        )
        self.calibrations = TradeNodeCalibrationStore(self.settings.trade_node_calibration_path)
        self.sessions = SessionStore()
        self.reranker = build_reranker(
            RerankerSettings(
                strategy=self.settings.reranker_strategy,
                model_path=self.settings.reranker_model_path,
                hf_home=self.settings.huggingface_home,
            )
        )
        self.llm_client = build_llm_client(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            model=self.settings.model,
            timeout=self.settings.request_timeout,
            ollama_base_url=self.settings.ollama_base_url,
            ollama_model=self.settings.ollama_model,
        )
        self._trade_plan_cache: dict[str, tuple[datetime, ChatPayload]] = {}

    def create_session(self) -> str:
        return self.sessions.create().session_id

    def list_sessions(self) -> list[dict[str, str]]:
        return [
            {
                "session_id": session.session_id,
                "title": session.title,
                "updated_at": session.updated_at,
            }
            for session in self.sessions.list_recent()
        ]

    def get_session_history(self, session_id: str) -> dict[str, object]:
        session = self.sessions.get(session_id)
        return {
            "session_id": session.session_id,
            "title": session.title,
            "messages": list(session.messages_for_ui),
        }

    def health(self) -> HealthResponse:
        stats = self.index.stats()
        return HealthResponse(
            status="ok",
            index_path=str(self.settings.index_path),
            articles=int(stats["articles"]),
            chunks=int(stats["chunks"]),
            vector_dim=int(stats["vector_dim"]),
            vector_backend=str(stats["vector_backend"]),
            vector_model=stats["vector_model"],
            model_mode=self.llm_client.mode,
            reranker_mode=self.reranker.mode,
        )

    async def stream_chat(self, session_id: str, message: str) -> AsyncIterator[str]:
        request_id = uuid4().hex
        try:
            session = self.sessions.get(session_id)
        except KeyError as exc:
            yield self._sse_event("error", {"message": str(exc), "request_id": request_id})
            return

        yield self._sse_event(
            "meta",
            {
                "request_id": request_id,
                "session_id": session_id,
                "model_mode": self.llm_client.mode,
                "reranker_mode": self.reranker.mode,
            },
        )

        try:
            payload = await self._build_payload(session=session, message=message)
            LOGGER.info(
                "chat_request_complete",
                extra={
                    "request_id": request_id,
                    "session_id": session_id,
                    "answer_mode": payload.answer_mode,
                    "model_mode": payload.session_state.model_mode,
                    "has_trade_plan": payload.trade_plan is not None,
                },
            )
            self._record_ui_history(session=session, user_message=message, payload=payload)
            for chunk in self._chunk_answer(payload.answer):
                yield self._sse_event("delta", {"text": chunk})
                await asyncio.sleep(0)
            yield self._sse_event("structured", payload.model_dump())
            yield self._sse_event("done", {"request_id": request_id})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "chat_request_failed",
                extra={"request_id": request_id, "session_id": session_id},
            )
            yield self._sse_event("error", {"message": str(exc), "request_id": request_id})

    def _record_ui_history(self, *, session: ChatSession, user_message: str, payload: ChatPayload) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not session.messages_for_ui:
            session.title = _session_title(user_message)
        session.updated_at = now
        session.messages_for_ui.append(
            {
                "id": f"user-{len(session.messages_for_ui)}",
                "role": "user",
                "text": user_message,
            }
        )
        assistant_message = payload.model_dump()
        assistant_message.update(
            {
                "id": f"assistant-{len(session.messages_for_ui)}",
                "role": "assistant",
                "text": payload.answer,
                "progressText": "",
            }
        )
        session.messages_for_ui.append(assistant_message)

    async def _build_payload(self, *, session: ChatSession, message: str) -> ChatPayload:
        self._compress_session(session)
        agent_plan = self.agent_planner.plan(message)
        if agent_plan.intent == "trade_plan":
            return await self._build_trade_plan_payload(session=session, agent_plan=agent_plan)

        stock_payload = await self._build_stock_node_payload(session=session, message=message)
        if stock_payload is not None:
            return stock_payload

        hits = self._retrieve_hits(message)
        citations = build_citations_from_hits(hits)
        weak_evidence = self._weak_evidence(hits)
        context = self._render_context(session)
        generation = await self.llm_client.generate(
            query=message,
            citations=citations,
            hits=hits,
            session_context=context,
            weak_evidence=weak_evidence,
        )
        disclaimer = generation.disclaimer or DEFAULT_DISCLAIMER

        session.turns.append(SessionTurn(role="user", content=message))
        session.turns.append(SessionTurn(role="assistant", content=generation.answer))
        self._compress_session(session)

        session_state = SessionState(
            session_id=session.session_id,
            turn_count=session.turn_count,
            summary=session.summary,
            model_mode=generation.mode,
        )
        return ChatPayload(
            answer=generation.answer,
            scenarios=generation.scenarios,
            citations=citations,
            disclaimer=disclaimer,
            session_state=session_state,
            answer_mode="generic_rag",
        )

    async def _build_trade_plan_payload(self, *, session: ChatSession, agent_plan: AgentPlan) -> ChatPayload:
        source_key = agent_plan.source_key or ""
        display_stock = agent_plan.display_stock or source_key.upper()
        cache_key = source_key.strip().lower()
        cached = self._get_cached_trade_plan(cache_key)
        if cached is not None:
            return self._record_cached_payload(session=session, query=agent_plan.raw_query, payload=cached)
        tools = await TradePlanToolRunner(
            market_data=self.market_data,
            fundamentals=self.fundamentals,
            quote_timeout=self.settings.quote_lookup_timeout,
            finnhub_api_key=self.settings.finnhub_api_key,
            quote_lookup_fn=self.quote_lookup,
        ).run(source_key)
        LOGGER.info(
            "trade_plan_tools_complete",
            extra={
                "source_key": source_key,
                "tools": tools.as_log_facts(),
            },
        )
        result = build_trade_plan(
            display_stock=display_stock,
            quote=tools.quote.data,
            candles=tools.candles.data or [],
            fundamentals=tools.fundamentals.data,
            market_context=tools.market_context.data,
            calibration=self.calibrations.get(source_key),
        )
        if result.trade_plan is None:
            session.turns.append(SessionTurn(role="user", content=agent_plan.raw_query))
            session.turns.append(SessionTurn(role="assistant", content=result.answer))
            self._compress_session(session)
            return ChatPayload(
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
        answer = result.answer
        scenarios = result.scenarios
        disclaimer = TRADE_PLAN_DISCLAIMER
        model_mode = "trade_plan_agent"
        answer_mode = "trade_plan_agent"
        if self.settings.enable_trade_plan_llm and self.llm_client.mode != "fallback":
            try:
                generation = await self.llm_client.generate_trade_plan_answer(
                    query=agent_plan.raw_query,
                    deterministic_answer=result.answer,
                    trade_facts=result.facts,
                    scenarios=result.scenarios,
                    session_context=self._render_context(session),
                )
                answer = generation.answer
                scenarios = generation.scenarios
                disclaimer = generation.disclaimer
                model_mode = generation.mode
                answer_mode = "trade_plan_agent_ai"
                if result.trade_plan is not None:
                    if not self._trade_plan_answer_is_consistent(answer, result.trade_plan):
                        raise ValueError("LLM trade plan answer failed deterministic fact validation")
                    if not _has_minimum_trade_plan_content(answer, result.trade_plan):
                        raise ValueError("LLM trade plan answer is too thin for user-facing response")
                    answer = self._ensure_trade_plan_sections(answer, result.trade_plan)
            except Exception:
                answer = result.answer
                scenarios = result.scenarios
                disclaimer = TRADE_PLAN_DISCLAIMER
                model_mode = "trade_plan_agent"
                answer_mode = "trade_plan_agent"

        session.turns.append(SessionTurn(role="user", content=agent_plan.raw_query))
        session.turns.append(SessionTurn(role="assistant", content=answer))
        self._compress_session(session)
        session_state = SessionState(
            session_id=session.session_id,
            turn_count=session.turn_count,
            summary=session.summary,
            model_mode=model_mode,
        )
        payload = ChatPayload(
            answer=answer,
            scenarios=scenarios,
            citations=[],
            disclaimer=disclaimer,
            session_state=session_state,
            answer_mode=answer_mode,
            trade_plan=result.trade_plan,
        )
        self._set_cached_trade_plan(cache_key, payload)
        return payload

    def _get_cached_trade_plan(self, cache_key: str) -> ChatPayload | None:
        item = self._trade_plan_cache.get(cache_key)
        if item is None:
            return None
        created_at, payload = item
        if datetime.now(timezone.utc) - created_at > timedelta(minutes=10):
            self._trade_plan_cache.pop(cache_key, None)
            return None
        return payload

    def _set_cached_trade_plan(self, cache_key: str, payload: ChatPayload) -> None:
        if not cache_key or payload.trade_plan is None:
            return
        self._trade_plan_cache[cache_key] = (datetime.now(timezone.utc), payload)

    def _record_cached_payload(self, *, session: ChatSession, query: str, payload: ChatPayload) -> ChatPayload:
        session.turns.append(SessionTurn(role="user", content=query))
        session.turns.append(SessionTurn(role="assistant", content=payload.answer))
        self._compress_session(session)
        cached_payload = payload.model_copy(deep=True)
        cached_payload.session_state = SessionState(
            session_id=session.session_id,
            turn_count=session.turn_count,
            summary=session.summary,
            model_mode=f"{payload.session_state.model_mode}:cached",
        )
        return cached_payload

    @staticmethod
    def _trade_plan_answer_is_consistent(answer: str, trade_plan) -> bool:
        normalized = answer.replace("\\r\\n", "\n").replace("\\n", "\n")
        if _has_bad_trade_plan_opening(normalized):
            return False
        if not _answer_prices_are_allowed(normalized, trade_plan):
            return False
        if _has_observation_location_contradiction(normalized, trade_plan):
            return False
        if _calls_defense_a_buy_point(normalized):
            return False
        if _has_defense_action_contradiction(normalized, trade_plan):
            return False
        current_price = float(trade_plan.current_price)
        location_terms = (
            "当前价格",
            "当前价",
            "价格目前",
            "目前价格",
            "现价",
        )
        in_range_terms = (
            "位于",
            "处于",
            "在",
            "落在",
        )
        for sentence in _split_sentences(normalized):
            if not any(term in sentence for term in location_terms):
                continue
            if not any(term in sentence for term in in_range_terms):
                continue
            for node in trade_plan.nodes:
                if not node.active or node.lower is None or node.upper is None:
                    continue
                lower = float(node.lower)
                upper = float(node.upper)
                if lower <= current_price <= upper:
                    continue
                range_tokens = {
                    f"{lower:.2f}-{upper:.2f}",
                    f"{lower:.2f} - {upper:.2f}",
                    f"{lower:.2f}到{upper:.2f}",
                    f"{lower:.2f} 至 {upper:.2f}",
                    f"{lower:.2f}至{upper:.2f}",
                }
                if any(token in sentence for token in range_tokens):
                    if any(
                        relation in sentence
                        for relation in (
                            "上方",
                            "下方",
                            "高于",
                            "低于",
                            "没到",
                            "尚未到",
                            "还在区间",
                            "已经低于",
                            "等待回到",
                        )
                    ):
                        continue
                    return False
        return True

    @staticmethod
    def _ensure_trade_plan_sections(answer: str, trade_plan) -> str:
        return answer

    async def _build_stock_node_payload(self, *, session: ChatSession, message: str) -> ChatPayload | None:
        if self.stock_nodes is None:
            return None
        stock_query = self.stock_nodes.parse_query(message)
        if stock_query is None:
            return None
        stock_records = self._stock_records_for_query(stock_query)
        wants_calculated_nodes = self.settings.enable_quote_lookup and calculated_nodes_requested(message)

        result = build_stock_node_answer(
            self.stock_nodes,
            stock_query,
            max_items=self.settings.stock_node_max_items,
        )
        if result is None and not wants_calculated_nodes:
            return None

        if result is None:
            answer = ""
            scenarios = []
            disclaimer = DEFAULT_DISCLAIMER
        else:
            answer, scenarios, disclaimer = result

        citations = []
        answer_mode = "stock_nodes"
        model_mode = "stock_nodes"

        calculated_answer = await self._build_calculated_node_answer(
            stock_query=stock_query,
            historical_answer=answer or None,
        ) if wants_calculated_nodes else None
        if calculated_answer is not None:
            answer, calculated_facts = calculated_answer
            answer_mode = "calculated_nodes"
            model_mode = "calculated_nodes"
        elif result is None:
            return None

        quote_requested = self.settings.enable_quote_lookup and quote_lookup_requested(message)
        quote_summary = None
        if quote_requested and answer_mode == "stock_nodes":
            quote = await self.quote_lookup(
                stock_query.source_key,
                timeout=self.settings.quote_lookup_timeout,
                provider=self.settings.quote_provider,
                api_key=self.settings.finnhub_api_key,
            )
            if quote is not None:
                quote_summary = build_quote_distance_summary(
                    quote,
                    stock_records,
                    max_nodes=self.settings.stock_node_max_items,
                )
        if quote_summary:
            answer = f"{answer}\n\n{quote_summary}"
        elif quote_requested:
            answer = f"{answer}\n\n实时行情工具暂时不可用，以下只列历史节点，不计算当前价距离。"

        hits = self._retrieve_stock_node_hits(stock_query, message)
        citations = build_citations_from_hits(hits)
        if self.llm_client.mode != "fallback":
            try:
                node_facts = self._stock_node_facts(stock_records)
                if calculated_answer is not None:
                    node_facts.append({"tool": "calculated_nodes", "facts": calculated_facts})
                if quote_summary:
                    node_facts.append({"tool": "quote_lookup", "summary": quote_summary})
                generation = await self.llm_client.generate_stock_node_answer(
                    query=message,
                    structured_answer=answer,
                    node_facts=node_facts,
                    citations=citations,
                    hits=hits,
                    session_context=self._render_context(session),
                )
                answer = generation.answer
                scenarios = generation.scenarios
                disclaimer = generation.disclaimer
                model_mode = generation.mode
                answer_mode = (
                    "calculated_nodes_ai"
                    if calculated_answer is not None
                    else "stock_nodes_ai"
                )
            except Exception:
                if calculated_answer is not None:
                    answer_mode = "calculated_nodes"
                    model_mode = "calculated_nodes"
                else:
                    answer_mode = "stock_nodes"
                    model_mode = "stock_nodes"

        session.turns.append(SessionTurn(role="user", content=message))
        session.turns.append(SessionTurn(role="assistant", content=answer))
        self._compress_session(session)

        session_state = SessionState(
            session_id=session.session_id,
            turn_count=session.turn_count,
            summary=session.summary,
            model_mode=model_mode,
        )
        return ChatPayload(
            answer=answer,
            scenarios=scenarios,
            citations=citations,
            disclaimer=disclaimer,
            session_state=session_state,
            answer_mode=answer_mode,
        )

    async def _build_calculated_node_answer(
        self,
        *,
        stock_query: StockNodeQuery,
        historical_answer: str | None,
    ) -> tuple[str, dict[str, object]] | None:
        quote = await self.quote_lookup(
            stock_query.source_key,
            timeout=self.settings.quote_lookup_timeout,
            provider="finnhub",
            api_key=self.settings.finnhub_api_key,
        )
        if quote is None:
            return None
        candles = await self.market_data.get_daily_candles(stock_query.source_key, lookback_days=420)
        plan = calculate_technical_node_plan(
            candles=candles,
            quote=quote,
            calibration=self.calibrations.get(stock_query.source_key),
        )
        if plan is None or not plan.data_sufficient:
            return None
        fundamentals, market_context = await asyncio.gather(
            self.fundamentals.get_snapshot(stock_query.source_key),
            build_market_context(self.market_data),
        )
        facts = plan.as_facts()
        facts["extra"] = calculated_node_extra_facts(
            fundamentals=fundamentals,
            market_context=market_context,
        )
        return (
            format_enhanced_calculated_node_answer(
                plan,
                historical_answer=historical_answer,
                fundamentals=fundamentals,
                market_context=market_context,
            ),
            facts,
        )

    def _stock_records_for_query(self, stock_query: StockNodeQuery) -> list[StockNodeRecord]:
        if stock_query.kind == "latest":
            latest = self.stock_nodes.query_latest(stock_query.source_key) if self.stock_nodes else None
            return [latest] if latest else []
        return self.stock_nodes.query_all(stock_query.source_key) if self.stock_nodes else []

    def _retrieve_stock_node_hits(self, stock_query: StockNodeQuery, message: str) -> list[SearchHit]:
        query = f"{stock_query.display_stock} {stock_query.source_key.upper()} {message}"
        return self._retrieve_hits(query)

    @staticmethod
    def _stock_node_facts(records: list[StockNodeRecord]) -> list[dict[str, object]]:
        return [
            {
                "stock": record.display_stock,
                "date": record.date,
                "nodes": record.nodes,
                "type": record.entry_type,
                "article_file": record.article_file,
                "summary": record.summary,
                "evidence": record.evidence,
            }
            for record in records[:12]
        ]

    def _retrieve_hits(self, query: str) -> list[SearchHit]:
        recalled = self.index.search(
            query,
            limit=self.settings.recall_limit,
            mode=self.settings.search_mode,
            dense_embedder=self.embedder,
        )
        if not recalled:
            return []
        return self.reranker.rerank(query, recalled, limit=self.settings.top_k)

    def _render_context(self, session: ChatSession) -> str:
        parts: list[str] = []
        if session.summary:
            parts.append(f"历史摘要：{session.summary}")
        for turn in session.turns[-self.settings.session_window :]:
            speaker = "用户" if turn.role == "user" else "助手"
            parts.append(f"{speaker}：{turn.content}")
        return "\n".join(parts).strip()

    def _compress_session(self, session: ChatSession) -> None:
        threshold = self.settings.summary_threshold
        if len(session.turns) <= threshold:
            return

        preserved = session.turns[-self.settings.session_window :]
        archived = session.turns[: -self.settings.session_window]
        summary_lines: list[str] = []
        if session.summary:
            summary_lines.append(session.summary)
        for turn in archived[-6:]:
            preview = turn.content.replace("\n", " ").strip()
            preview = preview[:100] + ("..." if len(preview) > 100 else "")
            speaker = "用户" if turn.role == "user" else "助手"
            summary_lines.append(f"{speaker}提到：{preview}")
        session.summary = "\n".join(summary_lines)[-1200:]
        session.turns = preserved

    def _weak_evidence(self, hits: list[SearchHit]) -> bool:
        if not hits:
            return True
        top_score = hits[0].rerank_score if hits[0].rerank_score is not None else hits[0].score
        return top_score < 0.15 or len(hits) < 2

    @staticmethod
    def _chunk_answer(answer: str, *, size: int = 24) -> list[str]:
        return [answer[index : index + size] for index in range(0, len(answer), size)] or [""]

    @staticmethod
    def _sse_event(event: str, payload: dict[str, object]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in "。！？!?\n":
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _session_title(message: str, *, limit: int = 24) -> str:
    title = " ".join(message.strip().split())
    if not title:
        return "新对话"
    return f"{title[:limit]}..." if len(title) > limit else title


def _answer_prices_are_allowed(answer: str, trade_plan) -> bool:
    allowed = _allowed_trade_plan_prices(trade_plan)
    if not allowed:
        return True
    for match in re.finditer(r"(?<![A-Za-z0-9])(\d{1,5}(?:\.\d{1,2})?)(?![A-Za-z0-9])", answer):
        if _ignored_numeric_context(answer, match):
            continue
        value = float(match.group(1))
        if not any(abs(value - allowed_value) <= 0.011 for allowed_value in allowed):
            return False
    return True


def _has_minimum_trade_plan_content(answer: str, trade_plan) -> bool:
    compact = answer.replace(" ", "")
    if len(compact) < 120:
        return False
    if f"{float(trade_plan.current_price):.2f}" not in compact:
        return False
    if not any(term in answer for term in ("观察区", "第一买入区", "买入区", "防守位")):
        return False
    if not any(term in answer for term in ("确认", "企稳", "失效", "风控", "风险")):
        return False

    active_nodes = [
        node
        for node in trade_plan.nodes
        if node.active and node.lower is not None and node.key in {"observation", "first_buy", "deep_buy", "defense"}
    ]
    if not active_nodes:
        return True
    return any(_node_price_is_mentioned(compact, node) for node in active_nodes)


def _node_price_is_mentioned(compact_answer: str, node) -> bool:
    lower = f"{float(node.lower):.2f}"
    if node.upper is None or float(node.upper) == float(node.lower):
        return lower in compact_answer
    upper = f"{float(node.upper):.2f}"
    return lower in compact_answer and upper in compact_answer


def _allowed_trade_plan_prices(trade_plan) -> set[float]:
    values = {round(float(trade_plan.current_price), 2)}
    for node in trade_plan.nodes:
        if not node.active:
            continue
        if node.lower is not None:
            values.add(round(float(node.lower), 2))
        if node.upper is not None:
            values.add(round(float(node.upper), 2))
    return values


def _ignored_numeric_context(text: str, match: re.Match[str]) -> bool:
    value = float(match.group(1))
    if value < 20 or value > 1000:
        return True
    start, end = match.span(1)
    before = text[max(0, start - 6) : start].upper()
    after = text[end : min(len(text), end + 3)]
    if any(indicator in before for indicator in ("MA", "RSI", "ATR", "EPS")):
        return True
    if after[:1] in {"%", "％", "日", "天", "周", "月", "年", "条", "次", "个"}:
        return True
    return False


def _has_bad_trade_plan_opening(answer: str) -> bool:
    first_sentence = _split_sentences(answer.strip())[0] if answer.strip() else ""
    if "？" in first_sentence or "?" in first_sentence:
        return True
    bad_fragments = (
        "可小仓位试探，还是先降风险",
        "可小仓试探，还是先降风险",
        "可小仓位试探，还是",
    )
    return any(fragment in answer[:120] for fragment in bad_fragments)


def _has_observation_location_contradiction(answer: str, trade_plan) -> bool:
    current_price = float(trade_plan.current_price)
    observation = next((node for node in trade_plan.nodes if node.key == "observation" and node.active), None)
    if observation is None or observation.lower is None or observation.upper is None:
        return False
    lower = float(observation.lower)
    upper = float(observation.upper)
    in_zone = lower <= current_price <= upper
    if in_zone:
        bad_fragments = (
            "只有价格进入该区域后",
            "只有价格进入观察区后",
            "等价格进入观察区",
            "等待价格回到观察区",
            "未进入前不追高",
        )
        return any(fragment in answer for fragment in bad_fragments)
    return False


def _calls_defense_a_buy_point(answer: str) -> bool:
    for sentence in _split_sentences(answer):
        if "防守位" not in sentence:
            continue
        if re.search(r"防守位[^。！？!?\n]{0,24}(是|作为|属于|当作)[^。！？!?\n]{0,12}(买入点|买点|买入区)", sentence):
            if "不是" not in sentence and "不属于" not in sentence:
                return True
    return False


def _has_defense_action_contradiction(answer: str, trade_plan) -> bool:
    if str(getattr(trade_plan, "action_state", "")) != "wait":
        return False
    defense = next((node for node in trade_plan.nodes if node.key == "defense" and node.active), None)
    if defense is None:
        return False
    for sentence in _split_sentences(answer):
        if "防守位" not in sentence and f"{float(defense.lower):.2f}" not in sentence:
            continue
        if "停止加仓" not in sentence and "降低风险" not in sentence:
            continue
        if any(term in sentence for term in ("未来", "买入后", "已经按计划买入", "已有仓位", "风控预案", "不是当前")):
            continue
        return True
    return False
