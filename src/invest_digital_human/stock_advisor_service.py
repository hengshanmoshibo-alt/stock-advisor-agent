from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator
from uuid import uuid4

from .chat_service import ChatSession, InvestmentChatService
from .conversation_intent_resolver import ConversationIntentResolver
from .llm_security_resolver import LLMSecurityResolver
from .schemas import ChatPayload
from .stock_advisor_explainer import StockAdvisorExplainer
from .stock_advisor_graph import LastTradeContext, StockAdvisorGraph, remember_trade_context


LOGGER = logging.getLogger(__name__)


class StockAdvisorService(InvestmentChatService):
    """Stock-advisor service backed by a LangGraph state graph."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.intent_resolver = ConversationIntentResolver(self.settings)
        self.security_resolver = LLMSecurityResolver(self.settings)
        self.explainer = StockAdvisorExplainer(self.settings)
        self._last_trade_context_by_session: dict[str, LastTradeContext] = {}
        self.graph = StockAdvisorGraph(
            service=self,
            intent_resolver=self.intent_resolver,
            explainer=self.explainer,
        )

    async def _build_payload(self, *, session: ChatSession, message: str) -> ChatPayload:
        self._compress_session(session)
        self.graph.intent_resolver = self.intent_resolver
        self.graph.explainer = self.explainer
        payload = await self.graph.invoke(
            session=session,
            message=message,
            last_trade_context=self._last_trade_context_by_session.get(session.session_id),
        )
        self._remember_trade_context(session, payload)
        return payload

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
                "vector_backend": getattr(self.embedder, "name", "none"),
                "reranker": getattr(self.reranker, "name", "none"),
            },
        )

        try:
            self.graph.intent_resolver = self.intent_resolver
            self.graph.explainer = self.explainer
            payload: ChatPayload | None = None
            async for event_type, event_payload in self.graph.stream(
                session=session,
                message=message,
                last_trade_context=self._last_trade_context_by_session.get(session.session_id),
            ):
                if event_type == "progress":
                    assert isinstance(event_payload, dict)
                    yield self._sse_event("progress", event_payload)
                    await asyncio.sleep(0)
                    continue
                if event_type == "payload":
                    assert isinstance(event_payload, ChatPayload)
                    payload = event_payload

            if payload is None:
                raise RuntimeError("StockAdvisorGraph did not produce a payload")

            self._remember_trade_context(session, payload)
            self._record_ui_history(session=session, user_message=message, payload=payload)
            for chunk in self._chunk_answer(payload.answer):
                yield self._sse_event("delta", {"text": chunk})
                await asyncio.sleep(0)
            yield self._sse_event("structured", payload.model_dump())
            yield self._sse_event("done", {"request_id": request_id})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "stock_advisor_graph_request_failed",
                extra={"request_id": request_id, "session_id": session_id},
            )
            yield self._sse_event("error", {"message": str(exc), "request_id": request_id})

    def _remember_trade_context(self, session: ChatSession, payload: ChatPayload) -> None:
        context = remember_trade_context(payload)
        if context is None:
            return
        self._last_trade_context_by_session[session.session_id] = context
