from __future__ import annotations

from .retrieval import SearchHit
from .text_utils import snippet


REALTIME_HINTS = (
    "指数或资产当前价格区间",
    "最近 1 到 5 个交易日走势",
    "最新财报、宏观或政策事件",
    "当前仓位、现金比例和可承受回撤",
)


def build_answer(query: str, hits: list[SearchHit]) -> str:
    if not hits:
        return (
            f"问题：{query}\n\n"
            "结论草稿：当前知识库里没有足够直接的历史证据支撑明确判断，"
            "暂时不建议给出确定性的买卖结论。\n\n"
            "下一步建议：把问题问得更具体，例如“是否适合抄底”“怎么看本轮回撤”"
            "“仓位该怎么控制”，这样更容易从历史文章中找到有效参考。"
        )

    top = hits[0].chunk
    evidence_lines = []
    for index, hit in enumerate(hits[:4], start=1):
        chunk = hit.chunk
        evidence_lines.append(
            f"{index}. [{chunk.published}]《{chunk.title}》\n"
            f"   片段：{snippet(chunk.text, limit=180)}\n"
            f"   来源：{chunk.url}"
        )

    return (
        f"问题：{query}\n\n"
        "结论草稿：\n"
        f"基于知识库中的历史文章，这个问题更适合先看风险与位置，再讨论进攻节奏。"
        f"当前最相关的历史参考来自《{top.title}》，但它依然只是历史观点整理，"
        "不等同于今天的实时判断。\n\n"
        "历史依据：\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        "当前仍需补充的实时信息：\n"
        + "\n".join(f"- {item}" for item in REALTIME_HINTS)
        + "\n\n风险提示：\n"
        "- 本回答只基于历史文章片段检索，不包含实时行情与新闻。\n"
        "- 如果要面向真实投资决策，必须把历史观点和当下市场事实严格区分。\n"
        "- 真正对外提供建议前，还需要补齐实时数据、风险约束和引用校验。"
    )
