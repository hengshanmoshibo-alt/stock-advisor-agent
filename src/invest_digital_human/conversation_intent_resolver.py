from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .config import AppSettings


ConversationIntentName = Literal[
    "trade_plan",
    "context_followup",
    "concept_explain",
    "article_history",
    "clarify",
]
ConversationRelation = Literal["followup", "new_stock", "none", "ambiguous"]
RequestedField = Literal[
    "watch_zone",
    "first_buy_zone",
    "deep_buy_zone",
    "defense",
    "confirmation",
    "invalidation",
    "rationale",
    "full_plan",
]


@dataclass(frozen=True, slots=True)
class LastTradeIntentContext:
    ticker: str
    display_stock: str
    current_price: float
    action_state: str
    confidence: str
    nodes: list[dict[str, Any]]
    confirmation_condition: str
    failure_condition: str


@dataclass(frozen=True, slots=True)
class ConversationIntent:
    intent: ConversationIntentName
    relation_to_last_trade: ConversationRelation
    target_ticker: str = ""
    target_name: str = ""
    requested_field: RequestedField | None = None
    confidence: float = 0.0
    clarifying_question: str = ""
    reason: str = ""


class ConversationIntentResolver:
    """LLM-only router for stock-advisor conversations.

    The model decides intent and context relation. This class only validates
    the JSON contract and safety boundaries before the service routes.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    async def resolve(
        self,
        query: str,
        *,
        last_trade_context: LastTradeIntentContext | None = None,
    ) -> ConversationIntent:
        raw = await self._call_model(query=query, last_trade_context=last_trade_context)
        payload = _loads_json_object(raw)
        return _coerce_intent(payload, has_last_trade_context=last_trade_context is not None)

    async def _call_model(
        self,
        *,
        query: str,
        last_trade_context: LastTradeIntentContext | None,
    ) -> str:
        if self.settings.api_key and self.settings.base_url and self.settings.model:
            try:
                return await self._call_openai_compatible(query, last_trade_context)
            except Exception:
                if self.settings.ollama_model and self.settings.ollama_model.lower() not in {"none", "off"}:
                    return await self._call_ollama(query, last_trade_context)
                raise
        if self.settings.ollama_model and self.settings.ollama_model.lower() not in {"none", "off"}:
            return await self._call_ollama(query, last_trade_context)
        raise RuntimeError("No LLM configured for conversation intent resolution")

    async def _call_openai_compatible(
        self,
        query: str,
        last_trade_context: LastTradeIntentContext | None,
    ) -> str:
        assert self.settings.api_key and self.settings.base_url and self.settings.model
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            response = await client.post(
                f"{self.settings.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": _user_prompt(query, last_trade_context)},
                    ],
                },
            )
            response.raise_for_status()
        choices = response.json().get("choices") or []
        return str(((choices[0] or {}).get("message") or {}).get("content") or "").strip()

    async def _call_ollama(
        self,
        query: str,
        last_trade_context: LastTradeIntentContext | None,
    ) -> str:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": self.settings.ollama_model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": _user_prompt(query, last_trade_context)},
                    ],
                },
            )
            response.raise_for_status()
        return str((response.json().get("message") or {}).get("content") or "").strip()


def _system_prompt() -> str:
    return (
        "你是股票建议系统的对话判断节点，只判断用户意图和上下文关系，不回答问题。"
        "你必须只输出严格 JSON，不要输出 Markdown、解释或多余文本。"
        "输出字段固定为：intent、relation_to_last_trade、target_ticker、target_name、requested_field、confidence、clarifying_question、reason。"
        "intent 只能是 trade_plan、context_followup、concept_explain、article_history、clarify。"
        "relation_to_last_trade 只能是 followup、new_stock、none、ambiguous。"
        "requested_field 只能是 watch_zone、first_buy_zone、deep_buy_zone、defense、confirmation、invalidation、rationale、full_plan 或 null。"
        "本系统当前只支持美股和美股 ETF。只有能确定为美股/美股 ETF 时，才允许 intent=trade_plan。"
        "如果用户问 A 股、港股、基金、加密货币、外汇，或者公司名明确但不是美股标的，intent=clarify，target_ticker 为空，clarifying_question 要说明当前只支持美股，并让用户输入美股名称或代码。"
        "如果用户给的是跨市场名称或市场不明确的公司名，不要猜 ticker；intent=clarify，并要求用户提供美股 ticker。"
        "如果用户问某只美股现在怎么买、买入节点、买点、建仓、加仓、防守位、止损、交易计划，intent=trade_plan，并输出 target_ticker。"
        "如果用户说换成 X、再看 X、X 呢，且 X 是明确美股或美股 ETF，intent=trade_plan，relation_to_last_trade=new_stock，并输出 target_ticker。"
        "如果用户没有说新股票，但明显是在追问上一轮交易计划里的字段，intent=context_followup，relation_to_last_trade=followup，并输出 requested_field。"
        "如果用户追问为什么不是现在买、为什么不能买、为什么只观察，requested_field 必须是 rationale，不要输出 first_buy_zone。"
        "如果有最近一次交易计划上下文，用户追问防守位、这个防守位是什么意思、止损线、跌破哪里失效，intent=context_followup，requested_field 必须是 defense 或 invalidation。"
        "只有在没有最近一次交易计划上下文时，防守位是什么意思才作为 concept_explain。"
        "如果用户问 MA200、RSI14、ATR、均线、观察区、第一买入区等概念是什么意思，intent=concept_explain。"
        "如果用户明确问历史文章、文章里、作者观点、以前提到过什么，intent=article_history。"
        "如果用户没有给股票且没有足够上下文，例如只说买入节点，intent=clarify，并给 clarifying_question。"
        "不要计算价格，不要生成买卖建议，不要编造 ticker。"
    )


def _user_prompt(query: str, last_trade_context: LastTradeIntentContext | None) -> str:
    context_json = (
        json.dumps(_context_to_json(last_trade_context), ensure_ascii=False, indent=2)
        if last_trade_context is not None
        else "null"
    )
    return (
        "请判断当前用户问题的路由，只输出 JSON。\n"
        f"当前用户问题：{query}\n\n"
        "最近一次成功交易计划上下文 JSON：\n"
        f"{context_json}\n\n"
        "必须输出单个 JSON 对象，不要输出数组。下面是示例，不能照抄示例：\n"
        '示例1：{"intent":"context_followup","relation_to_last_trade":"followup","target_ticker":"NVDA","target_name":"英伟达","requested_field":"first_buy_zone","confidence":0.93,"clarifying_question":"","reason":"用户追问上一轮交易计划的第一买点"}\n'
        '示例2：{"intent":"trade_plan","relation_to_last_trade":"new_stock","target_ticker":"MSFT","target_name":"微软","requested_field":"full_plan","confidence":0.94,"clarifying_question":"","reason":"用户切换到新的股票微软"}\n'
        '示例3：{"intent":"concept_explain","relation_to_last_trade":"none","target_ticker":"","target_name":"","requested_field":null,"confidence":0.95,"clarifying_question":"","reason":"用户询问技术指标概念"}\n'
        '示例4：{"intent":"clarify","relation_to_last_trade":"ambiguous","target_ticker":"","target_name":"","requested_field":null,"confidence":0.9,"clarifying_question":"请补充股票名称或代码，例如：微软现在买入节点是多少？","reason":"用户没有提供具体股票"}\n'
        '示例5：{"intent":"context_followup","relation_to_last_trade":"followup","target_ticker":"NVDA","target_name":"英伟达","requested_field":"rationale","confidence":0.93,"clarifying_question":"","reason":"用户追问为什么不是现在买"}\n'
        '示例6：{"intent":"context_followup","relation_to_last_trade":"followup","target_ticker":"NVDA","target_name":"英伟达","requested_field":"defense","confidence":0.93,"clarifying_question":"","reason":"用户追问防守位"}\n'
        '示例7：{"intent":"context_followup","relation_to_last_trade":"followup","target_ticker":"AMZN","target_name":"亚马逊","requested_field":"defense","confidence":0.93,"clarifying_question":"","reason":"用户在上一轮交易计划后追问防守位是什么意思"}\n'
        '示例8：{"intent":"clarify","relation_to_last_trade":"ambiguous","target_ticker":"","target_name":"中国银行","requested_field":null,"confidence":0.9,"clarifying_question":"当前股票建议项目只支持美股和美股 ETF。中国银行不是当前支持范围，请输入美股名称或代码，例如：微软、英伟达、AMZN、TSLA。","reason":"用户询问非美股标的"}'
    )


def _context_to_json(context: LastTradeIntentContext | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "ticker": context.ticker,
        "display_stock": context.display_stock,
        "current_price": context.current_price,
        "action_state": context.action_state,
        "confidence": context.confidence,
        "nodes": context.nodes,
        "confirmation_condition": context.confirmation_condition,
        "failure_condition": context.failure_condition,
    }


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Conversation intent resolver did not return a JSON object")
    return payload


def _coerce_intent(payload: dict[str, Any], *, has_last_trade_context: bool) -> ConversationIntent:
    intent = str(payload.get("intent") or "clarify").strip().lower()
    if intent not in {"trade_plan", "context_followup", "concept_explain", "article_history", "clarify"}:
        intent = "clarify"

    relation = str(payload.get("relation_to_last_trade") or "none").strip().lower()
    if relation not in {"followup", "new_stock", "none", "ambiguous"}:
        relation = "none"

    target_ticker = str(payload.get("target_ticker") or "").strip().upper()
    target_name = str(payload.get("target_name") or "").strip()
    requested_field = _coerce_requested_field(payload.get("requested_field"))
    confidence = _coerce_confidence(payload.get("confidence"))
    clarifying_question = str(payload.get("clarifying_question") or "").strip()
    reason = str(payload.get("reason") or "").strip()

    if confidence < 0.45:
        return _clarify("我还不能稳定判断你的问题，请补充股票名称或具体想问的内容。", confidence, reason)
    if intent == "context_followup" and not has_last_trade_context:
        return _clarify("我还没有上一只股票的交易计划，请先告诉我要分析哪只股票。", confidence, reason)
    if intent == "context_followup" and requested_field is None:
        requested_field = "full_plan"
    if intent == "trade_plan":
        if not _looks_like_ticker(target_ticker):
            return _clarify(
                clarifying_question
                or "当前股票建议项目只支持美股和美股 ETF。请补充明确的美股名称或代码，例如：微软、英伟达、AMZN、TSLA。",
                confidence,
                reason,
            )
        if not target_name:
            target_name = target_ticker
    if intent in {"concept_explain", "article_history", "clarify"}:
        target_ticker = ""
        target_name = ""
        if intent == "clarify" and not clarifying_question:
            clarifying_question = "请补充美股名称或代码，或者说明你想了解的概念。当前股票建议项目只支持美股和美股 ETF。"

    return ConversationIntent(
        intent=intent,  # type: ignore[arg-type]
        relation_to_last_trade=relation,  # type: ignore[arg-type]
        target_ticker=target_ticker,
        target_name=target_name,
        requested_field=requested_field,
        confidence=confidence,
        clarifying_question=clarifying_question,
        reason=reason,
    )


def _coerce_requested_field(value: Any) -> RequestedField | None:
    if value is None:
        return None
    field = str(value).strip().lower()
    allowed = {
        "watch_zone",
        "first_buy_zone",
        "deep_buy_zone",
        "defense",
        "confirmation",
        "invalidation",
        "rationale",
        "full_plan",
    }
    if field not in allowed:
        return None
    return field  # type: ignore[return-value]


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _looks_like_ticker(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", value.strip()))


def _clarify(question: str, confidence: float, reason: str) -> ConversationIntent:
    return ConversationIntent(
        intent="clarify",
        relation_to_last_trade="ambiguous",
        confidence=confidence,
        clarifying_question=question,
        reason=reason,
    )
