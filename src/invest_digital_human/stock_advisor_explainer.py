from __future__ import annotations

import httpx

from .config import AppSettings


CONCEPT_DISCLAIMER = "以上内容仅用于股票概念学习和研究参考，不构成投资建议。"


class StockAdvisorExplainer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    async def explain(self, query: str) -> str:
        if self.settings.api_key and self.settings.base_url and self.settings.model:
            try:
                return await self._openai_compatible(query)
            except Exception:
                if self.settings.ollama_model and self.settings.ollama_model.lower() not in {"none", "off"}:
                    return await self._ollama(query)
                raise
        if self.settings.ollama_model and self.settings.ollama_model.lower() not in {"none", "off"}:
            return await self._ollama(query)
        return "当前没有可用的大模型，不能解释这个概念。"

    async def _openai_compatible(self, query: str) -> str:
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
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": query},
                    ],
                },
            )
            response.raise_for_status()
        choices = response.json().get("choices") or []
        return str(((choices[0] or {}).get("message") or {}).get("content") or "").strip()

    async def _ollama(self, query: str) -> str:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": self.settings.ollama_model,
                    "stream": False,
                    "options": {"temperature": 0.2},
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": query},
                    ],
                },
            )
            response.raise_for_status()
        return str((response.json().get("message") or {}).get("content") or "").strip()


def _system_prompt() -> str:
    return (
        "你是面向普通人的股票学习助手。回答要短、清楚、少用术语。"
        "如果用户问技术指标或交易概念，用生活化语言解释：它是什么、怎么看、常见误区。"
        "不要编造实时行情，不要给具体买卖建议，不要假装已经计算了某只股票。"
        "如果用户问某只股票买入节点，提醒需要走交易计划计算链路。"
        "回答使用简体中文，尽量控制在 5 到 8 句话。"
    )
