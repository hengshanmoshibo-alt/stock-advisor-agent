from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from .retrieval import SearchHit
from .schemas import Citation, ScenarioCard
from .text_utils import snippet


SCENARIO_META = {
    "bullish": "乐观情景",
    "neutral": "中性情景",
    "bearish": "谨慎情景",
}

TIMING_KEYWORDS = {
    "买入",
    "买点",
    "节点",
    "建仓",
    "加仓",
    "介入",
    "上车",
    "卖出",
    "止盈",
    "止损",
    "仓位",
}

DEFAULT_DISCLAIMER = (
    "风险提示：本回答仅基于历史文章观点整理，不构成实时投资建议。"
    "市场会变化，请结合你的期限、仓位、风险承受能力和最新信息独立判断。"
)


TRADE_PLAN_DISCLAIMER = (
    "风险提示：本回答基于行情、K线、基本面和大盘环境的规则化计算结果，"
    "只用于交易计划参考，不构成投资建议。市场会变化，请结合你的期限、仓位和风险承受能力独立判断。"
)


@dataclass(slots=True)
class LLMGeneration:
    answer: str
    scenarios: list[ScenarioCard]
    disclaimer: str
    mode: str


class BaseLLMClient:
    mode = "fallback"

    async def generate(
        self,
        *,
        query: str,
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
        weak_evidence: bool,
    ) -> LLMGeneration:
        raise NotImplementedError

    async def generate_stock_node_answer(
        self,
        *,
        query: str,
        structured_answer: str,
        node_facts: list[dict[str, Any]],
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
    ) -> LLMGeneration:
        return LLMGeneration(
            answer=structured_answer,
            scenarios=_coerce_scenarios(None, hits, citations),
            disclaimer=DEFAULT_DISCLAIMER,
            mode=self.mode,
        )

    async def generate_trade_plan_answer(
        self,
        *,
        query: str,
        deterministic_answer: str,
        trade_facts: dict[str, Any],
        scenarios: list[ScenarioCard],
        session_context: str,
    ) -> LLMGeneration:
        return LLMGeneration(
            answer=deterministic_answer,
            scenarios=scenarios,
            disclaimer=TRADE_PLAN_DISCLAIMER,
            mode=self.mode,
        )


class OpenAICompatibleLLMClient(BaseLLMClient):
    mode = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        *,
        query: str,
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
        weak_evidence: bool,
    ) -> LLMGeneration:
        prompt = _render_user_prompt(
            query=query,
            citations=citations,
            hits=hits,
            session_context=session_context,
            weak_evidence=weak_evidence,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0.25,
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
        raw_content = _extract_openai_content(response.json())
        return _parse_generation(
            raw_content,
            query=query,
            hits=hits,
            citations=citations,
            weak_evidence=weak_evidence,
            mode=self.mode,
        )

    async def generate_stock_node_answer(
        self,
        *,
        query: str,
        structured_answer: str,
        node_facts: list[dict[str, Any]],
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
    ) -> LLMGeneration:
        prompt = _render_stock_node_prompt(
            query=query,
            structured_answer=structured_answer,
            node_facts=node_facts,
            citations=citations,
            hits=hits,
            session_context=session_context,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": _stock_node_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
        raw_content = _extract_openai_content(response.json())
        return _parse_stock_node_generation(
            raw_content,
            fallback_answer=structured_answer,
            hits=hits,
            citations=citations,
            mode=f"{self.mode}:stock_nodes",
        )

    async def generate_trade_plan_answer(
        self,
        *,
        query: str,
        deterministic_answer: str,
        trade_facts: dict[str, Any],
        scenarios: list[ScenarioCard],
        session_context: str,
    ) -> LLMGeneration:
        prompt = _render_trade_plan_prompt(
            query=query,
            deterministic_answer=deterministic_answer,
            trade_facts=trade_facts,
            scenarios=scenarios,
            session_context=session_context,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0.15,
                    "messages": [
                        {"role": "system", "content": _trade_plan_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
        raw_content = _extract_openai_content(response.json())
        return _parse_trade_plan_generation(
            raw_content,
            fallback_scenarios=scenarios,
            mode=f"{self.mode}:trade_plan",
        )


class OllamaLLMClient(BaseLLMClient):
    mode = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        *,
        query: str,
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
        weak_evidence: bool,
    ) -> LLMGeneration:
        prompt = _render_user_prompt(
            query=query,
            citations=citations,
            hits=hits,
            session_context=session_context,
            weak_evidence=weak_evidence,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.25},
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
        payload = response.json()
        raw_content = str((payload.get("message") or {}).get("content") or "").strip()
        if not raw_content:
            raise ValueError("Ollama response did not contain message content")
        return _parse_generation(
            raw_content,
            query=query,
            hits=hits,
            citations=citations,
            weak_evidence=weak_evidence,
            mode=self.mode,
        )

    async def generate_stock_node_answer(
        self,
        *,
        query: str,
        structured_answer: str,
        node_facts: list[dict[str, Any]],
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
    ) -> LLMGeneration:
        prompt = _render_stock_node_prompt(
            query=query,
            structured_answer=structured_answer,
            node_facts=node_facts,
            citations=citations,
            hits=hits,
            session_context=session_context,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.2},
                    "messages": [
                        {"role": "system", "content": _stock_node_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
        payload = response.json()
        raw_content = str((payload.get("message") or {}).get("content") or "").strip()
        if not raw_content:
            raise ValueError("Ollama response did not contain message content")
        return _parse_stock_node_generation(
            raw_content,
            fallback_answer=structured_answer,
            hits=hits,
            citations=citations,
            mode=f"{self.mode}:stock_nodes",
        )

    async def generate_trade_plan_answer(
        self,
        *,
        query: str,
        deterministic_answer: str,
        trade_facts: dict[str, Any],
        scenarios: list[ScenarioCard],
        session_context: str,
    ) -> LLMGeneration:
        prompt = _render_trade_plan_prompt(
            query=query,
            deterministic_answer=deterministic_answer,
            trade_facts=trade_facts,
            scenarios=scenarios,
            session_context=session_context,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.15},
                    "messages": [
                        {"role": "system", "content": _trade_plan_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
        payload = response.json()
        raw_content = str((payload.get("message") or {}).get("content") or "").strip()
        if not raw_content:
            raise ValueError("Ollama response did not contain message content")
        return _parse_trade_plan_generation(
            raw_content,
            fallback_scenarios=scenarios,
            mode=f"{self.mode}:trade_plan",
        )


class FailoverLLMClient(BaseLLMClient):
    """Try a local model first, then an external compatible model if it fails."""

    mode = "failover"

    def __init__(self, *, primary: BaseLLMClient, fallback: BaseLLMClient) -> None:
        self.primary = primary
        self.fallback = fallback

    async def generate(
        self,
        *,
        query: str,
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
        weak_evidence: bool,
    ) -> LLMGeneration:
        try:
            return await self.primary.generate(
                query=query,
                citations=citations,
                hits=hits,
                session_context=session_context,
                weak_evidence=weak_evidence,
            )
        except Exception:
            return await self.fallback.generate(
                query=query,
                citations=citations,
                hits=hits,
                session_context=session_context,
                weak_evidence=weak_evidence,
            )

    async def generate_stock_node_answer(
        self,
        *,
        query: str,
        structured_answer: str,
        node_facts: list[dict[str, Any]],
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
    ) -> LLMGeneration:
        try:
            return await self.primary.generate_stock_node_answer(
                query=query,
                structured_answer=structured_answer,
                node_facts=node_facts,
                citations=citations,
                hits=hits,
                session_context=session_context,
            )
        except Exception:
            return await self.fallback.generate_stock_node_answer(
                query=query,
                structured_answer=structured_answer,
                node_facts=node_facts,
                citations=citations,
                hits=hits,
                session_context=session_context,
            )

    async def generate_trade_plan_answer(
        self,
        *,
        query: str,
        deterministic_answer: str,
        trade_facts: dict[str, Any],
        scenarios: list[ScenarioCard],
        session_context: str,
    ) -> LLMGeneration:
        try:
            return await self.primary.generate_trade_plan_answer(
                query=query,
                deterministic_answer=deterministic_answer,
                trade_facts=trade_facts,
                scenarios=scenarios,
                session_context=session_context,
            )
        except Exception:
            return await self.fallback.generate_trade_plan_answer(
                query=query,
                deterministic_answer=deterministic_answer,
                trade_facts=trade_facts,
                scenarios=scenarios,
                session_context=session_context,
            )


class FallbackLLMClient(BaseLLMClient):
    mode = "fallback"

    async def generate(
        self,
        *,
        query: str,
        citations: list[Citation],
        hits: list[SearchHit],
        session_context: str,
        weak_evidence: bool,
    ) -> LLMGeneration:
        if not hits:
            answer = (
                f"关于“{query}”，当前知识库里没有找到足够直接的历史证据。"
                "更稳妥的做法是先明确你的期限、仓位和风险底线，再继续追问。"
            )
            answer = _ensure_actionable_answer(
                answer,
                query=query,
                hits=hits,
                citations=citations,
                weak_evidence=True,
            )
            return LLMGeneration(
                answer=answer,
                scenarios=_coerce_scenarios(None, hits, citations),
                disclaimer=DEFAULT_DISCLAIMER,
                mode=self.mode,
            )

        top = hits[0]
        answer = (
            f"如果只基于当前召回到的历史文章，“{query}”更适合先按审慎偏积极的方式理解。"
            f"最相关的依据来自《{top.chunk.title}》({top.chunk.published})，"
            "它反复强调先看条件是否成立，再决定是否提高仓位或延长持有。"
        )
        if weak_evidence:
            answer += " 但这次证据集中度一般，所以更适合作为观察框架，不适合当成高确定性结论。"
        answer = _ensure_actionable_answer(
            answer,
            query=query,
            hits=hits,
            citations=citations,
            weak_evidence=weak_evidence,
        )
        return LLMGeneration(
            answer=answer,
            scenarios=_coerce_scenarios(None, hits, citations),
            disclaimer=DEFAULT_DISCLAIMER,
            mode=self.mode,
        )


def build_llm_client(
    *,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    timeout: float,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
) -> BaseLLMClient:
    normalized_ollama_model = (ollama_model or "").strip()
    external_client: BaseLLMClient | None = None
    if api_key and base_url and model:
        external_client = OpenAICompatibleLLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
    if normalized_ollama_model and normalized_ollama_model.lower() not in {"none", "off"}:
        ollama_client = OllamaLLMClient(
            base_url=ollama_base_url or "http://127.0.0.1:11434",
            model=normalized_ollama_model,
            timeout=timeout,
        )
        if external_client is not None:
            return FailoverLLMClient(primary=ollama_client, fallback=external_client)
        return ollama_client
    if external_client is not None:
        return external_client
    return FallbackLLMClient()


def _system_prompt() -> str:
    return (
        "你是一个中文投资数字人，只能依据提供的历史文章证据回答。"
        "所有字段都必须使用简体中文。"
        "不要假装知道实时行情，也不要编造证据。"
        "如果证据偏弱，要明确写出条件不足、适用前提和主要风险。"
        "如果用户在问买入节点、建仓时机、加仓位置、卖出节点这类时机问题，"
        "你必须在 answer 里给出至少 3 条具体、可执行的条件化节点，每条都包含："
        "触发条件、动作建议、失效条件。"
        "这些节点必须是条件化描述，不能伪装成实时价格指令。"
        "你必须输出严格 JSON，对象结构如下："
        '{"answer":"主回答","disclaimer":"风险提示","scenarios":['
        '{"key":"bullish","title":"乐观情景","stance":"...","reasoning":"...","risk":"..."},'
        '{"key":"neutral","title":"中性情景","stance":"...","reasoning":"...","risk":"..."},'
        '{"key":"bearish","title":"谨慎情景","stance":"...","reasoning":"...","risk":"..."}'
        "]}"
    )


def _render_user_prompt(
    *,
    query: str,
    citations: list[Citation],
    hits: list[SearchHit],
    session_context: str,
    weak_evidence: bool,
) -> str:
    evidence_blocks = []
    for index, hit in enumerate(hits[:5], start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[证据{index}] 标题：{hit.chunk.title}",
                    f"日期：{hit.chunk.published}",
                    f"链接：{hit.chunk.url}",
                    f"检索分数：{hit.score:.3f}",
                    (
                        f"重排分数：{hit.rerank_score:.3f}"
                        if hit.rerank_score is not None
                        else "重排分数：未启用"
                    ),
                    f"摘录：{snippet(hit.chunk.text, limit=260)}",
                ]
            )
        )

    citation_lines = [
        f"- {citation.title} | {citation.published} | {citation.score:.3f} | {citation.snippet}"
        for citation in citations
    ]

    extra_rule = ""
    if _is_timing_query(query):
        extra_rule = (
            "这是一个明确的时机/节点问题。"
            "answer 必须写出“具体节点：”并列出至少 3 条节点。"
            "每条节点都要写成“当……时，做……；如果……失效，就……”的格式。"
            "节点必须能落地到分批建仓、等待突破确认、回踩不破、缩量企稳、放量续强、跌破失效这些动作层面。"
        )

    return "\n\n".join(
        [
            f"用户问题：{query}",
            f"证据强弱：{'偏弱' if weak_evidence else '尚可'}",
            f"会话上下文：\n{session_context or '无'}",
            "引用摘要：\n" + ("\n".join(citation_lines) if citation_lines else "无"),
            "详细证据：\n" + ("\n\n".join(evidence_blocks) if evidence_blocks else "无"),
            (
                "请先给结论，再给三种情景分析。"
                "回答必须明确写明这是基于历史文章观点整理，不是实时盘面判断。"
            ),
            extra_rule or "如果证据不足，就明确告诉用户哪些条件需要继续观察。",
        ]
    )


def _extract_openai_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LLM response did not include choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return str(content)


def _parse_generation(
    raw_content: str,
    *,
    query: str,
    hits: list[SearchHit],
    citations: list[Citation],
    weak_evidence: bool,
    mode: str,
) -> LLMGeneration:
    payload = _extract_json(raw_content)
    scenarios = _coerce_scenarios(payload.get("scenarios"), hits, citations)
    answer = _sanitize_answer(str(payload.get("answer") or "").strip())
    if not answer:
        raise ValueError("LLM response did not contain answer text")
    answer = _ensure_actionable_answer(
        answer,
        query=query,
        hits=hits,
        citations=citations,
        weak_evidence=weak_evidence,
    )
    disclaimer = _normalize_disclaimer(str(payload.get("disclaimer") or "").strip())
    return LLMGeneration(
        answer=answer,
        scenarios=scenarios,
        disclaimer=disclaimer,
        mode=mode,
    )


def _parse_stock_node_generation(
    raw_content: str,
    *,
    fallback_answer: str,
    hits: list[SearchHit],
    citations: list[Citation],
    mode: str,
) -> LLMGeneration:
    payload = _extract_json(raw_content)
    answer = _sanitize_answer(str(payload.get("answer") or "").strip()) or fallback_answer
    disclaimer = _normalize_disclaimer(str(payload.get("disclaimer") or "").strip())
    return LLMGeneration(
        answer=answer,
        scenarios=_coerce_scenarios(payload.get("scenarios"), hits, citations),
        disclaimer=disclaimer,
        mode=mode,
    )


def _parse_trade_plan_generation(
    raw_content: str,
    *,
    fallback_scenarios: list[ScenarioCard],
    mode: str,
) -> LLMGeneration:
    payload = _extract_json(raw_content)
    answer = _sanitize_answer(str(payload.get("answer") or "").strip())
    if not answer:
        raise ValueError("LLM response did not contain trade plan answer text")
    disclaimer = _normalize_trade_plan_disclaimer(str(payload.get("disclaimer") or "").strip())
    scenarios = _coerce_trade_plan_scenarios(payload.get("scenarios"), fallback_scenarios)
    return LLMGeneration(
        answer=answer,
        scenarios=scenarios,
        disclaimer=disclaimer,
        mode=mode,
    )


def _trade_plan_system_prompt() -> str:
    return (
        "交易计划回答必须让普通投资者看得懂，不要堆指标，不要写成长篇研报。"
        "先给一句话结论，再给“现在怎么做”“关键价位”“为什么”“什么时候作废”。"
        "回答里可以保留价格和节点，但不要解释 MA20/MA50/RSI/ATR 的专业定义，除非用一句白话说明。"
        "交易计划回答必须按以下小标题输出：当前结论、现在怎么做、关键价位、为什么这么判断、什么时候作废、风险提示。"
        "每个节点都必须保留确定性工具给出的价格、公式和激活状态；如果节点未激活，也必须明确写出未激活和原因。"
        "观察区只能解释为观察和等待企稳，不要写成买入触发；第一买入区才是可考虑分批试探的位置。"
        "如果当前价已经位于观察区内，必须说“已经在观察区内，仍需等待企稳”，不要说“等价格回到观察区”。"
        "防守位必须解释为已有仓位的风险线或计划失效线，不要写成买入点。"
        "你是中文交易计划解释助手。你只能基于用户提供的确定性工具结果改写交易计划，"
        "不得自行计算、修改或新增任何价格、日期、区间、指标、行动状态或计划等级。"
        "不得输出确定性工具 facts 和确定性答案里没有的任何价格；不能把历史会话里其它股票的价格带入当前回答。"
        "关键价位必须逐字使用当前 facts 的节点价格，不要四舍五入，不要改写成近似整数。"
        "不得引用历史文章，不得声称访问了额外行情、财报或新闻。"
        "如果工具结果显示数据不足，必须明确说明数据不足，不能编造买入节点。"
        "输出必须是严格 JSON，结构为："
        '{"answer":"中文主回答","disclaimer":"风险提示","scenarios":['
        '{"key":"bullish","title":"积极情景","stance":"...","reasoning":"...","risk":"..."},'
        '{"key":"neutral","title":"中性情景","stance":"...","reasoning":"...","risk":"..."},'
        '{"key":"bearish","title":"谨慎情景","stance":"...","reasoning":"...","risk":"..."}'
        "]}"
    )


def _render_trade_plan_prompt(
    *,
    query: str,
    deterministic_answer: str,
    trade_facts: dict[str, Any],
    scenarios: list[ScenarioCard],
    session_context: str,
) -> str:
    scenario_facts = [scenario.model_dump() for scenario in scenarios]
    required_sections = (
        "answer 必须使用普通人能看懂的表达，避免堆指标。"
        "answer 必须包含这些小标题：当前结论、现在怎么做、关键价位、为什么这么判断、什么时候作废、风险提示。"
        "关键价位部分必须覆盖观察区、第一买入区、深度买入区和防守位；未激活的节点要明确说暂不参考。"
        "关键价位的小节只能使用当前 facts JSON 的节点价格，不能使用会话上下文里其它股票的价格。"
        "观察区只表示观察，不代表可以买；第一买入区才表示可以考虑分批试探。"
        "如果 current_price 已在观察区 lower/upper 之间，必须写成“当前价已经在观察区内，仍需等待企稳”。"
        "防守位是风险控制线，不是买入价，也不是预测底部。"
        "如果不确定某个价位是否属于当前 facts，就不要写这个价位。"
        "第一段必须直接说：现在是先等、可小仓试探、可加仓，还是先降风险。"
        "如果 facts 里的某个节点 active=false，仍然必须在对应小标题下写明“未激活”，"
        "并使用 facts 中的 action/formula 解释原因。"
    )
    deterministic_answer = required_sections + "\n\n" + deterministic_answer
    return "\n\n".join(
        [
            f"用户问题：{query}",
            f"会话上下文：\n{session_context or '无'}",
            "确定性交易计划答案：\n" + deterministic_answer,
            "确定性工具 facts JSON：\n" + json.dumps(trade_facts, ensure_ascii=False, indent=2),
            "确定性情景卡 JSON：\n" + json.dumps(scenario_facts, ensure_ascii=False, indent=2),
            (
                "请把确定性交易计划改写成更自然、更像 Agent 的中文回答，但主要面向普通用户。"
                "不要把全部专业指标堆给用户；专业指标只用来解释“为什么”，最多列 3 条关键原因。"
                "必须保留所有价格节点、公式来源、行动状态、计划等级、确认条件、失效条件和风险提示。"
                "你可以调整表达顺序和语言，但不能新增任何 facts 中没有的价格、日期、指标或结论。"
                "不要把会话上下文里上一只股票的价格复用到当前股票；当前股票的价格只能来自本轮 facts JSON。"
                "回答必须说明这是基于行情、K线、基本面和大盘环境的规则化交易计划，不构成投资建议。"
            ),
        ]
    )


def _coerce_trade_plan_scenarios(
    payload: Any,
    fallback_scenarios: list[ScenarioCard],
) -> list[ScenarioCard]:
    if not isinstance(payload, list):
        return fallback_scenarios

    by_key: dict[str, dict[str, Any]] = {}
    for item in payload:
        if isinstance(item, dict) and item.get("key") in SCENARIO_META:
            by_key[str(item["key"])] = item

    if not by_key:
        return fallback_scenarios

    fallback_by_key = {scenario.key: scenario for scenario in fallback_scenarios}
    result: list[ScenarioCard] = []
    for key in ("bullish", "neutral", "bearish"):
        item = by_key.get(key)
        fallback = fallback_by_key.get(key)
        if item is None:
            if fallback is not None:
                result.append(fallback)
            continue
        result.append(
            ScenarioCard(
                key=key,  # type: ignore[arg-type]
                title=str(item.get("title") or (fallback.title if fallback else SCENARIO_META[key])).strip(),
                stance=str(item.get("stance") or (fallback.stance if fallback else "")).strip(),
                reasoning=str(item.get("reasoning") or (fallback.reasoning if fallback else "")).strip(),
                risk=str(item.get("risk") or (fallback.risk if fallback else "")).strip(),
            )
        )
    return result or fallback_scenarios


def _stock_node_system_prompt() -> str:
    return (
        "你是一个中文投资资料整理助手。你只能依据用户提供的结构化节点和文章证据回答。"
        "价格、日期、文件名必须来自结构化节点，不能编造或补充不存在的价位。"
        "文章证据只能用于解释背景和语气，不得覆盖结构化节点。"
        "输出必须是严格 JSON："
        '{"answer":"主回答","disclaimer":"风险提示","scenarios":['
        '{"key":"bullish","title":"积极情景","stance":"...","reasoning":"...","risk":"..."},'
        '{"key":"neutral","title":"中性情景","stance":"...","reasoning":"...","risk":"..."},'
        '{"key":"bearish","title":"谨慎情景","stance":"...","reasoning":"...","risk":"..."}'
        "]}"
    )


def _render_stock_node_prompt(
    *,
    query: str,
    structured_answer: str,
    node_facts: list[dict[str, Any]],
    citations: list[Citation],
    hits: list[SearchHit],
    session_context: str,
) -> str:
    evidence_blocks = []
    for index, hit in enumerate(hits[:4], start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[文章证据{index}] 标题：{hit.chunk.title}",
                    f"日期：{hit.chunk.published}",
                    f"来源：{hit.chunk.source_path}",
                    f"摘录：{snippet(hit.chunk.text, limit=260)}",
                ]
            )
        )
    citation_lines = [
        f"- {citation.title} | {citation.published} | {citation.snippet}"
        for citation in citations
    ]
    return "\n\n".join(
        [
            f"用户问题：{query}",
            f"会话上下文：\n{session_context or '无'}",
            "结构化节点事实 JSON：\n" + json.dumps(node_facts, ensure_ascii=False, indent=2),
            "确定性节点答案：\n" + structured_answer,
            "文章引用摘要：\n" + ("\n".join(citation_lines) if citation_lines else "无"),
            "文章证据片段：\n" + ("\n\n".join(evidence_blocks) if evidence_blocks else "无"),
            (
                "请把结构化节点组织成更自然的中文回答。"
                "第一段解释这些节点呈现出的时间脉络；第二段列出关键节点，必须保留日期、价格节点和文件名；"
                "最后提醒这是历史文章整理，不是实时建议。"
                "不要加入结构化节点里没有的价格、日期或文件名。"
            ),
        ]
    )


def _extract_json(raw_content: str) -> dict[str, Any]:
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        return json.loads(content[start : end + 1])


def _coerce_scenarios(
    payload: Any,
    hits: list[SearchHit],
    citations: list[Citation],
) -> list[ScenarioCard]:
    normalized: dict[str, Any] = {}
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("key") in SCENARIO_META:
                normalized[str(item["key"])] = item
    elif isinstance(payload, dict):
        for key in SCENARIO_META:
            if key in payload:
                normalized[key] = payload[key]

    top_title = hits[0].chunk.title if hits else "未检索到强相关文章"
    top_date = hits[0].chunk.published if hits else "日期未知"
    top_snippet = citations[0].snippet if citations else "当前没有足够直接的文章摘录可引用。"

    defaults = {
        "bullish": {
            "title": "乐观情景",
            "stance": "只有在历史文章强调的积极条件继续成立时，才适合更偏进攻地理解当前机会。",
            "reasoning": (
                f"当前最相关的文章是《{top_title}》({top_date})。"
                "如果其中提到的趋势延续、风险偏好恢复、核心逻辑没有被破坏，"
                "那么更积极的仓位或更长的持有周期才有依据。"
            ),
            "risk": "一旦关键前提失效，乐观判断会很快退化，追高和放大仓位都容易吃回撤。",
        },
        "neutral": {
            "title": "中性情景",
            "stance": "把它当成条件观察题，而不是立即下重注的确定性机会。",
            "reasoning": (
                "现有证据更适合支持“边观察边验证”的中性框架。"
                f"当前最能落地的依据是这段摘录：{top_snippet}"
            ),
            "risk": "如果一直停留在模糊判断里，容易既错过趋势确认，又在震荡里反复交易。",
        },
        "bearish": {
            "title": "谨慎情景",
            "stance": "如果核心证据不足，优先防守，而不是把历史观点误当成当前确定性信号。",
            "reasoning": (
                "历史文章能提供方法和框架，但不能替代实时市场验证。"
                "当证据集中度不够或风险条件变化时，收缩仓位、延后决策通常更合理。"
            ),
            "risk": "过度谨慎会错过向上行情，但在证据不足时，控制回撤仍然优先于追求短期收益。",
        },
    }

    result: list[ScenarioCard] = []
    for key, title in SCENARIO_META.items():
        item = normalized.get(key)
        if not isinstance(item, dict):
            item = {}
        base = defaults[key]
        result.append(
            ScenarioCard(
                key=key,  # type: ignore[arg-type]
                title=str(item.get("title") or title or base["title"]).strip(),
                stance=str(item.get("stance") or base["stance"]).strip(),
                reasoning=str(item.get("reasoning") or base["reasoning"]).strip(),
                risk=str(item.get("risk") or base["risk"]).strip(),
            )
        )
    return result


def build_citations_from_hits(hits: list[SearchHit], *, limit: int = 4) -> list[Citation]:
    citations: list[Citation] = []
    for hit in hits[:limit]:
        citations.append(
            Citation(
                title=hit.chunk.title,
                published=hit.chunk.published,
                url=hit.chunk.url,
                snippet=snippet(hit.chunk.text, limit=140),
                score=round(hit.rerank_score if hit.rerank_score is not None else hit.score, 3),
            )
        )
    return citations


def _normalize_disclaimer(value: str) -> str:
    if not value:
        return DEFAULT_DISCLAIMER
    chinese_count = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    ascii_letters = sum(1 for char in value if char.isascii() and char.isalpha())
    if chinese_count == 0 or ascii_letters > chinese_count:
        return DEFAULT_DISCLAIMER
    return value


def _normalize_trade_plan_disclaimer(value: str) -> str:
    if not value:
        return TRADE_PLAN_DISCLAIMER
    chinese_count = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    ascii_letters = sum(1 for char in value if char.isascii() and char.isalpha())
    if chinese_count == 0 or ascii_letters > chinese_count:
        return TRADE_PLAN_DISCLAIMER
    return value


def _sanitize_answer(answer: str) -> str:
    answer = answer.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    markers = [
        "\n\n**disclaimer:**",
        "\n\n**scenarios:**",
        "\n```json",
        '\n{"answer"',
        '\n{\n    "answer"',
    ]
    cleaned = answer
    for marker in markers:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].rstrip()
    return cleaned


def _is_timing_query(query: str) -> bool:
    return any(keyword in query for keyword in TIMING_KEYWORDS)


def _ensure_actionable_answer(
    answer: str,
    *,
    query: str,
    hits: list[SearchHit],
    citations: list[Citation],
    weak_evidence: bool,
) -> str:
    if not _is_timing_query(query):
        return answer
    if "具体节点" in answer and "1." in answer:
        return answer
    if "具体节点" in answer:
        base = answer.rstrip()
    else:
        base = answer.rstrip() + "\n\n具体节点："
    return base + "\n" + _build_timing_nodes(
        hits=hits,
        citations=citations,
        weak_evidence=weak_evidence,
    )


def _build_timing_nodes(
    *,
    hits: list[SearchHit],
    citations: list[Citation],
    weak_evidence: bool,
) -> str:
    top_title = hits[0].chunk.title if hits else "历史文章"
    top_date = hits[0].chunk.published if hits else "日期未知"
    ref = f"主要参考《{top_title}》({top_date}) 这一类历史观点。"

    if weak_evidence:
        return "\n".join(
            [
                f"1. 只在突破近阶段震荡上沿后第一次回踩不破时，先试探建仓 20% 到 30%；如果回踩后重新跌回震荡区，就撤回试单。{ref}",
                "2. 只在财报、行业催化或核心利好落地后出现放量续强时，再加第二笔；如果只是脉冲拉升但量能跟不上，就不要追。",
                "3. 只在回撤到关键中期支撑后出现缩量止跌、次日转强时，再考虑分批补仓；如果支撑被有效跌破，就继续等待。",
            ]
        )

    anchor_snippet = citations[0].snippet if citations else "当前证据集中在趋势延续、耐心持有和风险控制。"
    return "\n".join(
        [
            f"1. 当价格放量突破前一轮核心震荡区，并且随后回踩不破时，可以先建第一笔 30%；如果突破后很快跌回箱体，就视为失效。{ref}",
            "2. 当趋势已经确认、但还处在主升浪早中段时，可以采用 30% 到 30% 到 40% 的分批建仓，而不是一次性梭哈；如果趋势开始高位放量滞涨，就暂停加仓。",
            f"3. 当关键催化兑现后仍能保持强势，并且市场没有出现明显转弱信号时，再加最后一笔；如果催化兑现后冲高回落、量价背离，就不再追加。参考摘录：{anchor_snippet}",
        ]
    )
