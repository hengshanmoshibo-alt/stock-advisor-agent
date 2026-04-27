from __future__ import annotations

import asyncio
import unittest

from invest_digital_human.llm_client import (
    BaseLLMClient,
    DEFAULT_DISCLAIMER,
    FailoverLLMClient,
    LLMGeneration,
    TRADE_PLAN_DISCLAIMER,
    build_llm_client,
    _parse_generation,
    _parse_trade_plan_generation,
    _render_trade_plan_prompt,
    _trade_plan_system_prompt,
)
from invest_digital_human.schemas import ScenarioCard


class TradePlanLLMPromptTest(unittest.TestCase):
    def test_trade_plan_prompt_contains_deterministic_facts(self) -> None:
        prompt = _render_trade_plan_prompt(
            query="AMD buy zone",
            deterministic_answer="AMD 交易计划\n当前价：319.00\n第一买入区：250.00-270.00",
            trade_facts={
                "snapshot": {"current_price": 319.0},
                "nodes": [{"key": "first_buy", "lower": 250.0, "upper": 270.0}],
            },
            scenarios=[
                ScenarioCard(
                    key="neutral",
                    title="中性情景",
                    stance="等待",
                    reasoning="进入区间再看",
                    risk="跌破防守位",
                )
            ],
            session_context="",
        )
        system_prompt = _trade_plan_system_prompt()

        self.assertIn("319.00", prompt)
        self.assertIn("250.0", prompt)
        self.assertIn("active=false", prompt)
        self.assertIn("观察区", prompt)
        self.assertIn("不得自行计算", system_prompt)
        self.assertIn("不得引用历史文章", system_prompt)

    def test_disclaimer_normalization_is_separated_by_mode(self) -> None:
        generic = _parse_generation(
            '{"answer":"普通回答","disclaimer":"","scenarios":[]}',
            query="普通问题",
            hits=[],
            citations=[],
            weak_evidence=False,
            mode="test",
        )
        trade_plan = _parse_trade_plan_generation(
            '{"answer":"交易计划回答","disclaimer":"","scenarios":[]}',
            fallback_scenarios=[],
            mode="test:trade_plan",
        )

        self.assertEqual(generic.disclaimer, DEFAULT_DISCLAIMER)
        self.assertEqual(trade_plan.disclaimer, TRADE_PLAN_DISCLAIMER)


class LLMFailoverTest(unittest.TestCase):
    def test_build_llm_client_prefers_local_with_external_fallback(self) -> None:
        client = build_llm_client(
            api_key="external-key",
            base_url="https://example.com/v1",
            model="external-model",
            timeout=1,
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="local-model",
        )

        self.assertIsInstance(client, FailoverLLMClient)

    def test_failover_client_uses_external_when_primary_fails(self) -> None:
        client = FailoverLLMClient(primary=RaisingLLMClient(), fallback=SuccessfulLLMClient())

        generation = asyncio.run(
            client.generate(
                query="MA200 是什么？",
                citations=[],
                hits=[],
                session_context="",
                weak_evidence=False,
            )
        )

        self.assertEqual(generation.mode, "external-test")
        self.assertEqual(generation.answer, "external answer")


class RaisingLLMClient(BaseLLMClient):
    mode = "raising-test"

    async def generate(self, **kwargs) -> LLMGeneration:  # type: ignore[no-untyped-def]
        raise RuntimeError("local model failed")


class SuccessfulLLMClient(BaseLLMClient):
    mode = "external-test"

    async def generate(self, **kwargs) -> LLMGeneration:  # type: ignore[no-untyped-def]
        return LLMGeneration(answer="external answer", scenarios=[], disclaimer="", mode=self.mode)


if __name__ == "__main__":
    unittest.main()
