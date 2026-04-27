from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import httpx

from .config import AppSettings


SecurityIntent = Literal["trade_plan", "article_history", "generic"]


@dataclass(frozen=True, slots=True)
class ResolvedSecurity:
    ticker: str
    display_name: str
    intent: SecurityIntent
    confidence: float
    reason: str = ""


class LLMSecurityResolver:
    """Resolve a user's current message into a security and intent.

    The LLM does semantic recognition, but this class still validates the
    output so an ambiguous prompt like "买入节点" cannot become a fabricated
    ticker.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    async def resolve(self, query: str) -> ResolvedSecurity | None:
        raw = await self._call_model(query)
        payload = _loads_json_object(raw)
        intent = str(payload.get("intent") or "generic").strip().lower()
        if intent not in {"trade_plan", "article_history", "generic"}:
            intent = "generic"
        ticker = str(payload.get("ticker") or "").strip().upper()
        display_name = str(payload.get("display_name") or ticker).strip()
        confidence = _coerce_confidence(payload.get("confidence"))
        reason = str(payload.get("reason") or "")

        if confidence < 0.45:
            return None
        if intent == "trade_plan":
            if not _looks_like_ticker(ticker):
                return ResolvedSecurity("", display_name, "generic", confidence, reason)
            if not _security_mentioned_in_query(query, ticker=ticker, display_name=display_name):
                return ResolvedSecurity("", display_name, "generic", confidence, reason)
        else:
            ticker = ""

        return ResolvedSecurity(
            ticker=ticker,
            display_name=display_name or ticker,
            intent=intent,  # type: ignore[arg-type]
            confidence=confidence,
            reason=reason,
        )

    async def _call_model(self, query: str) -> str:
        if self.settings.ollama_model and self.settings.ollama_model.lower() not in {"none", "off"}:
            try:
                return await self._call_ollama(query)
            except Exception:
                if self.settings.api_key and self.settings.base_url and self.settings.model:
                    return await self._call_openai_compatible(query)
                raise
        if self.settings.api_key and self.settings.base_url and self.settings.model:
            return await self._call_openai_compatible(query)
        raise RuntimeError("No LLM configured for security resolution")

    async def _call_openai_compatible(self, query: str) -> str:
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
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": _user_prompt(query)},
                    ],
                },
            )
            response.raise_for_status()
        choices = response.json().get("choices") or []
        return str(((choices[0] or {}).get("message") or {}).get("content") or "").strip()

    async def _call_ollama(self, query: str) -> str:
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
                        {"role": "user", "content": _user_prompt(query)},
                    ],
                },
            )
            response.raise_for_status()
        return str((response.json().get("message") or {}).get("content") or "").strip()


def _system_prompt() -> str:
    return (
        "你是证券意图识别器，只做分类和证券识别，不回答投资问题。"
        "必须输出严格 JSON，不要输出 Markdown。"
        "输出字段固定为：ticker、display_name、intent、confidence、reason。"
        "intent 只能是 trade_plan、article_history、generic。"
        "只有当用户明确提到某只股票或 ETF，并询问买入节点、买点、建仓、加仓、防守位、止损、交易计划、现在能不能买时，intent 才能是 trade_plan。"
        "如果用户说 再看X、换成X、改成X、看看X，并且 X 是明确股票或 ETF，也视为交易计划切换请求，intent=trade_plan。"
        "display_name 必须优先使用用户原文里出现的证券名称，例如用户说 Intel 就写 Intel，用户说 英特尔 就写 英特尔。"
        "如果用户只问 MA200、RSI、ATR、均线、观察区、第一买入区等概念，intent=generic，ticker 为空。"
        "如果用户没有给出明确证券，例如只说 买入节点，intent=generic，ticker 为空。"
        "如果用户明确问历史文章、文章里、作者观点、提到过什么，intent=article_history。"
        "常见证券别名参考：微软=MSFT，英伟达=NVDA，英特尔=INTC，联合健康=UNH，谷歌=GOOGL，台积电=TSM，AMD=AMD。"
        "不要根据上一轮对话补全股票，只根据当前这一句话判断。"
        "不要把指标名、普通英文词或行业词当作 ticker。"
    )


def _user_prompt(query: str) -> str:
    return (
        "识别下面问题的证券和意图，只输出 JSON：\n"
        f"{query}\n"
        '示例1：{"ticker":"INTC","display_name":"英特尔","intent":"trade_plan","confidence":0.95,"reason":"用户明确询问英特尔买入节点"}\n'
        '示例2：{"ticker":"","display_name":"","intent":"generic","confidence":0.95,"reason":"用户询问 MA200 概念，不是交易计划"}\n'
        '示例3：{"ticker":"","display_name":"","intent":"generic","confidence":0.90,"reason":"用户没有提供具体证券"}\n'
        '示例4：{"ticker":"MSFT","display_name":"微软","intent":"trade_plan","confidence":0.95,"reason":"用户问现在买入微软怎么样"}\n'
        '示例5：{"ticker":"UNH","display_name":"联合健康","intent":"trade_plan","confidence":0.95,"reason":"用户询问联合健康买入节点"}'
    )


def _loads_json_object(raw: str) -> dict[str, object]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM security resolver did not return a JSON object")
    return payload


def _looks_like_ticker(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", value))


def _security_mentioned_in_query(query: str, *, ticker: str, display_name: str) -> bool:
    normalized_query = query.strip().lower()
    normalized_ticker = ticker.strip().lower()
    normalized_display = display_name.strip().lower()
    if normalized_ticker and re.search(rf"(?<![a-z0-9]){re.escape(normalized_ticker)}(?![a-z0-9])", normalized_query):
        return True
    return bool(normalized_display and normalized_display in normalized_query)


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
