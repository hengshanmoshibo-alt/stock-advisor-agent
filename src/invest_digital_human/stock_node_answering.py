from __future__ import annotations

from .schemas import ScenarioCard
from .stock_nodes import StockNodeKnowledgeBase, StockNodeQuery, StockNodeRecord


STOCK_NODE_DISCLAIMER = (
    "风险提示：本回答只基于历史文章中已经整理出的买入、加仓、建仓或防守节点，"
    "不包含实时行情，也不构成投资建议。"
)


def build_stock_node_answer(
    knowledge_base: StockNodeKnowledgeBase,
    query: StockNodeQuery,
    *,
    max_items: int = 8,
) -> tuple[str, list[ScenarioCard], str] | None:
    if query.kind == "latest":
        latest = knowledge_base.query_latest(query.source_key)
        if latest is None:
            return None
        answer = _format_latest_answer(query.display_stock, latest)
    elif query.kind == "exists":
        latest = knowledge_base.query_latest(query.source_key)
        if latest is None:
            return None
        answer = _format_exists_answer(query.display_stock, latest)
    else:
        records = knowledge_base.query_all(query.source_key)
        if not records:
            return None
        answer = _format_all_answer(query.display_stock, records, max_items=max_items)
    return answer, _stock_node_scenarios(query.display_stock), STOCK_NODE_DISCLAIMER


def _format_all_answer(stock: str, records: list[StockNodeRecord], *, max_items: int) -> str:
    shown = records[:max_items]
    dates = [record.date for record in records if record.date]
    if dates:
        summary = f"{stock} 在这批文章里有 {len(records)} 条明确买入、加仓或防守节点，时间范围是 {dates[-1]} 到 {dates[0]}。"
    else:
        summary = f"{stock} 在这批文章里有 {len(records)} 条明确买入、加仓或防守节点。"
    if len(records) > max_items:
        summary += f" 以下先列最近 {max_items} 条，另有 {len(records) - max_items} 条更早记录。"

    lines = [summary, "", "主要节点："]
    lines.extend(_format_record_line(record) for record in shown)
    return "\n".join(lines)


def _format_latest_answer(stock: str, record: StockNodeRecord) -> str:
    return "\n".join(
        [
            f"{stock} 最近一次明确节点是 {record.date}。",
            "",
            "节点：",
            _format_record_line(record),
        ]
    )


def _format_exists_answer(stock: str, record: StockNodeRecord) -> str:
    return f"有。{stock} 最近一次明确节点是：{_format_record_line(record)}"


def _format_record_line(record: StockNodeRecord) -> str:
    nodes = " / ".join(record.nodes) if record.nodes else "未列出具体价格"
    entry_type = record.entry_type or "明确节点"
    return f"- {record.date}：{nodes}；{entry_type}；{record.article_file}"


def _stock_node_scenarios(stock: str) -> list[ScenarioCard]:
    return [
        ScenarioCard(
            key="bullish",
            title="积极情景",
            stance=f"{stock} 的历史节点可作为分批观察参考。",
            reasoning="如果后续走势重新接近这些历史买入或防守区域，可以结合当时文章语境复核是否仍有参考价值。",
            risk="这些节点来自历史文章，不代表当前实时价格或新的买入建议。",
        ),
        ScenarioCard(
            key="neutral",
            title="中性情景",
            stance="更适合把节点当作历史决策记录，而不是直接交易指令。",
            reasoning="同一股票在不同日期可能出现不同价位，需要按时间和市场背景分开理解。",
            risk="只看价格节点而忽略仓位、现金流和宏观环境，容易误读作者原意。",
        ),
        ScenarioCard(
            key="bearish",
            title="谨慎情景",
            stance="如果当前市场环境已明显变化，历史买点应降低权重。",
            reasoning="结构化节点只能说明文章当时明确写过什么，不能替代最新财报、估值和风险条件。",
            risk="把历史防守位当成无条件买点，可能放大回撤。",
        ),
    ]
