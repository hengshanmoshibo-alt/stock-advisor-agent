from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .stock_nodes import StockNodeKnowledgeBase


AgentIntent = Literal["trade_plan", "technical_nodes", "market_explain", "stock_nodes", "generic_rag"]

TRADE_PLAN_TOOLS = (
    "quote_lookup",
    "market_candles",
    "technical_nodes",
    "fundamentals",
    "market_context",
)

ARTICLE_KEYWORDS = ("文章", "历史", "提到", "作者", "这批文章", "历史文章")
TRADE_KEYWORDS = (
    "怎么买",
    "买入节点",
    "买点",
    "建仓",
    "加仓",
    "防守",
    "失效",
    "止损",
    "能不能买",
    "能不能建仓",
    "现在买",
    "当前买",
    "第一买点",
    "交易计划",
    "节点",
    "entry",
    "buy zone",
)
MARKET_EXPLAIN_KEYWORDS = ("怎么看", "为什么", "回调", "大盘", "市场")
CHINESE_SOURCE_ALIASES = {
    "微软": "msft",
    "英伟达": "nvda",
    "苹果": "aapl",
    "谷歌": "googl",
    "亚马逊": "amzn",
    "特斯拉": "tsla",
    "沃尔玛": "wmt",
    "台积电": "tsm",
    "阿里": "baba",
    "拼多多": "pdd",
    "理想": "li",
    "联合健康": "unh",
    "联合健康集团": "unh",
    "unitedhealth": "unh",
}


@dataclass(frozen=True, slots=True)
class AgentPlan:
    intent: AgentIntent
    source_key: str | None = None
    display_stock: str | None = None
    tools: tuple[str, ...] = ()
    use_articles: bool = False
    raw_query: str = ""


class AgentPlanner:
    def __init__(self, stock_nodes: StockNodeKnowledgeBase | None) -> None:
        self.stock_nodes = stock_nodes

    def plan(self, query: str) -> AgentPlan:
        source_key = self._match_source_key(query)
        display_stock = self.stock_nodes.display_name(source_key) if self.stock_nodes and source_key else None
        if source_key and self._is_article_stock_query(query):
            return AgentPlan(
                intent="stock_nodes",
                source_key=source_key,
                display_stock=display_stock,
                tools=("article_rag",),
                use_articles=True,
                raw_query=query,
            )
        if source_key and self._is_trade_plan_query(query):
            return AgentPlan(
                intent="trade_plan",
                source_key=source_key,
                display_stock=display_stock,
                tools=TRADE_PLAN_TOOLS,
                use_articles=False,
                raw_query=query,
            )
        if self._is_market_explain_query(query):
            return AgentPlan(intent="market_explain", raw_query=query)
        return AgentPlan(intent="generic_rag", raw_query=query)

    def _match_source_key(self, query: str) -> str | None:
        normalized = query.strip().lower()
        for alias, source_key in CHINESE_SOURCE_ALIASES.items():
            if alias.lower() in normalized:
                return source_key
        if self.stock_nodes is not None:
            source_key = self.stock_nodes._match_source_key(query)  # noqa: SLF001
            if source_key:
                return source_key
        match = re.search(r"(?<![a-z0-9])([a-z]{1,5})(?![a-z0-9])", normalized)
        return match.group(1) if match else None

    @staticmethod
    def _is_article_stock_query(query: str) -> bool:
        return any(keyword in query for keyword in ARTICLE_KEYWORDS)

    @staticmethod
    def _is_trade_plan_query(query: str) -> bool:
        normalized = query.strip().lower()
        return any(keyword in normalized for keyword in TRADE_KEYWORDS)

    @staticmethod
    def _is_market_explain_query(query: str) -> bool:
        return any(keyword in query for keyword in MARKET_EXPLAIN_KEYWORDS)
