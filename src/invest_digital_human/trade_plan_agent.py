from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .calibration import TradeNodeCalibration
from .fundamental_data import FundamentalSnapshot
from .market_context import MarketContext
from .market_data import MarketCandle
from .quote_lookup import QuoteSnapshot
from .schemas import (
    ScenarioCard,
    TradePlanBacktest,
    TradePlanBacktestSummary,
    TradePlanCalibration,
    TradePlanMetrics,
    TradePlanNode,
    TradePlanPayload,
    TradePlanScoreBreakdown,
    TradePlanVolume,
)
from .technical_nodes import CalculatedNode, calculate_technical_node_plan
from .trade_scoring import ScoreBreakdown, score_trade_plan


ActionState = Literal["wait", "starter_allowed", "add_allowed", "risk_reduce"]
Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class TradePlanResult:
    answer: str
    action_state: ActionState
    confidence: Confidence
    facts: dict[str, object]
    scenarios: list[ScenarioCard]
    trade_plan: TradePlanPayload | None = None


def build_trade_plan(
    *,
    display_stock: str,
    quote: QuoteSnapshot | None,
    candles: list[MarketCandle],
    fundamentals: FundamentalSnapshot | None,
    market_context: MarketContext | None,
    calibration: TradeNodeCalibration | None = None,
) -> TradePlanResult:
    if quote is None:
        return _insufficient_result(display_stock, reason="缺少实时价格，无法计算交易节点。")
    plan = calculate_technical_node_plan(candles=candles, quote=quote, calibration=calibration)
    if plan is None or not plan.data_sufficient:
        return _insufficient_result(display_stock, reason="K线数据不足，不能生成完整交易计划。")

    score_breakdown = score_trade_plan(
        trend_stage=plan.trend_stage,
        weak_state=plan.snapshot.weak_state,
        rsi14=plan.snapshot.rsi14,
        volume=plan.volume_profile,
        fundamentals=fundamentals,
        market_context=market_context,
    )
    if plan.calibration is not None and plan.calibration.confidence_adjustment:
        score_breakdown = ScoreBreakdown(
            technical=score_breakdown.technical,
            volume=score_breakdown.volume,
            fundamentals=score_breakdown.fundamentals,
            market=score_breakdown.market,
            event_risk=score_breakdown.event_risk,
            total=score_breakdown.total + plan.calibration.confidence_adjustment,
            reasons=[
                *score_breakdown.reasons,
                f"校准置信度修正：{plan.calibration.confidence_adjustment:+d}",
            ],
        )
    action_state, confidence, reasons = _classify_plan(
        plan=plan,
        fundamentals=fundamentals,
        score_breakdown=score_breakdown,
    )
    nodes_by_key = {node.key: node for node in plan.nodes}
    facts = plan.as_facts()
    facts["action_state"] = action_state
    facts["confidence"] = confidence
    facts["fundamentals"] = fundamentals.as_facts() if fundamentals else None
    facts["market_context"] = market_context.as_facts() if market_context else None
    facts["score_breakdown"] = score_breakdown.as_facts()
    facts["parameter_source"] = plan.parameter_source
    facts["calibration"] = plan.calibration.as_facts() if plan.calibration else None
    facts["backtest_summary"] = plan.backtest_summary.as_facts()

    trade_plan = _build_structured_trade_plan(
        display_stock=display_stock or quote.ticker,
        facts=facts,
        action_state=action_state,
        confidence=confidence,
        nodes_by_key=nodes_by_key,
        risk_adjustments=reasons,
        fundamentals=fundamentals,
        market_context=market_context,
        score_breakdown=score_breakdown,
    )
    answer = _render_answer(
        display_stock=display_stock or quote.ticker,
        plan=plan,
        nodes_by_key=nodes_by_key,
        action_state=action_state,
        confidence=confidence,
        reasons=reasons,
        fundamentals=fundamentals,
        market_context=market_context,
    )
    return TradePlanResult(
        answer=answer,
        action_state=action_state,
        confidence=confidence,
        facts=facts,
        scenarios=_trade_plan_scenarios(display_stock or quote.ticker, action_state, confidence),
        trade_plan=trade_plan,
    )


def _render_answer(
    *,
    display_stock: str,
    plan,
    nodes_by_key: dict[str, CalculatedNode],
    action_state: ActionState,
    confidence: Confidence,
    reasons: list[str],
    fundamentals: FundamentalSnapshot | None,
    market_context: MarketContext | None,
) -> str:
    return _render_plain_answer(
        display_stock=display_stock,
        plan=plan,
        nodes_by_key=nodes_by_key,
        action_state=action_state,
        confidence=confidence,
        reasons=reasons,
        fundamentals=fundamentals,
        market_context=market_context,
    )

    lines = [
        f"{display_stock} 交易计划，价格节点由系统确定性计算。",
        f"当前价：{plan.snapshot.current_price:.2f}，数据日期：{plan.snapshot.as_of}",
        f"行动状态：{_action_label(action_state)}，计划等级：{_confidence_label(confidence)}",
        f"参数来源：{plan.parameter_source}",
        "",
        "关键指标：",
        (
            f"- MA20={_fmt(plan.snapshot.ma20)}，MA50={_fmt(plan.snapshot.ma50)}，"
            f"MA200={_fmt(plan.snapshot.ma200)}，RSI14={_fmt(plan.snapshot.rsi14)}，ATR14={_fmt(plan.snapshot.atr14)}"
        ),
        (
            f"- 60日高/低={_fmt(plan.snapshot.high_60)}/{_fmt(plan.snapshot.low_60)}，"
            f"120日高/低={_fmt(plan.snapshot.high_120)}/{_fmt(plan.snapshot.low_120)}"
        ),
        "",
        "交易节点：",
    ]
    lines.extend(_node_lines(nodes_by_key))
    lines.extend(
        [
            "",
            "确认条件：",
            f"- {_confirmation_condition(action_state)}",
            "",
            "失效条件：",
            f"- {_failure_condition(nodes_by_key)}",
            "",
            "风险修正：",
        ]
    )
    lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(_fundamental_lines(fundamentals))
    lines.extend(_market_lines(market_context))
    lines.append("")
    lines.append("风险提示：以上是基于价格、K线、量能、基本面和市场环境的规则化交易计划，不构成投资建议。")
    return "\n".join(lines)


def _render_plain_answer(
    *,
    display_stock: str,
    plan,
    nodes_by_key: dict[str, CalculatedNode],
    action_state: ActionState,
    confidence: Confidence,
    reasons: list[str],
    fundamentals: FundamentalSnapshot | None,
    market_context: MarketContext | None,
) -> str:
    current_price = plan.snapshot.current_price
    lines = [
        f"结论：{display_stock} 现在{_plain_action_label(action_state)}",
        (
            f"当前价 {current_price:.2f}，计划等级是“{_confidence_label(confidence)}”。"
            "价格节点由行情和K线规则计算。"
        ),
        "",
        "现在怎么做：",
        f"- {_plain_position_instruction(nodes_by_key, current_price)}",
        f"- {_plain_overlap_instruction(nodes_by_key)}",
        "",
        "关键价格：",
    ]
    lines.extend(_compact_price_lines(nodes_by_key, current_price))
    lines.extend(
        [
            "",
            "触发条件：",
            f"- {_confirmation_condition(action_state)}",
            "",
            "风险预案：",
            f"- {_failure_condition(nodes_by_key)}",
            "",
            "主要原因：",
        ]
    )
    lines.extend(_plain_reason_lines(plan=plan, reasons=reasons, fundamentals=fundamentals, market_context=market_context, limit=3))
    lines.extend(
        [
            "",
            f"一句话：{_one_line_summary(nodes_by_key, current_price)}",
            "以上只是规则化交易计划，不构成投资建议。",
        ]
    )
    return "\n".join(lines)


def _plain_action_label(action_state: ActionState) -> str:
    return {
        "wait": "先等，不急着买",
        "starter_allowed": "可以小仓位试探，但必须分批",
        "add_allowed": "可以按计划加仓，但仍要守住风控线",
        "risk_reduce": "先控制风险，不适合继续加仓",
    }[action_state]


def _plain_action_instruction(
    action_state: ActionState,
    current_price: float,
    nodes_by_key: dict[str, CalculatedNode],
) -> str:
    if action_state == "wait":
        observation = nodes_by_key.get("observation")
        if observation is not None and observation.lower <= current_price <= observation.upper:
            return "现在已经在观察区内，但还不是系统认可的主动买入点；要先看能不能企稳。"
        return "现在不是系统认可的主动买入点，先等价格回到关键区间并企稳。"
    if action_state == "starter_allowed":
        return "只适合小仓位试探，不适合一次性重仓。"
    if action_state == "add_allowed":
        return "可以按节点分批加仓，但每一笔都要有失效条件。"
    return "先减小风险敞口，等重新站稳后再看。"


def _plain_position_instruction(nodes_by_key: dict[str, CalculatedNode], current_price: float) -> str:
    first_buy = nodes_by_key.get("first_buy")
    observation = nodes_by_key.get("observation")
    if first_buy is not None:
        if current_price > first_buy.upper:
            return f"更合适的第一买入区在 {_node_value(first_buy)}，当前价还在区间上方，没到之前不要追高。"
        if first_buy.lower <= current_price <= first_buy.upper:
            return f"当前价已经进入第一买入区 {_node_value(first_buy)}，但仍要等企稳确认，不是自动买入。"
        return f"当前价已经低于第一买入区 {_node_value(first_buy)}，先看能否止跌收回，不急于接下跌。"
    if observation is not None:
        if observation.lower <= current_price <= observation.upper:
            return f"当前价已经在观察区 {_node_value(observation)} 内，意思是先看企稳，不是立刻买。"
        return f"目前只给观察区 {_node_value(observation)}，意思是先看，不代表到了就立刻买。"
    return "当前数据没有给出可执行买入区，先不做交易动作。"


def _plain_failure_instruction(nodes_by_key: dict[str, CalculatedNode]) -> str:
    defense = nodes_by_key.get("defense")
    if defense is not None:
        return (
            f"防守位 {_node_value(defense)} 是未来买入后的风控预案，不是当前买点；"
            "如果未来已经按计划买入，有效跌破后停止加仓或降低风险。"
        )
    observation = nodes_by_key.get("observation")
    if observation is not None:
        return f"如果跌破观察区下沿 {observation.lower:.2f} 后收不回来，这个计划就先失效。"
    return "没有防守位时，不执行买入计划。"


def _plain_node_lines(nodes_by_key: dict[str, CalculatedNode]) -> list[str]:
    labels = {
        "observation": "观察区",
        "first_buy": "第一买入区",
        "deep_buy": "深度买入区",
        "defense": "防守位",
    }
    explanations = {
        "observation": "先观察，等企稳确认，不是自动买。",
        "first_buy": "回撤压力减弱后，才考虑分批试探。",
        "deep_buy": "只有更深回调且止跌后才考虑。",
        "defense": "未来买入后的风控预案，不是买入点。",
    }
    lines: list[str] = []
    for key in ("observation", "first_buy", "deep_buy", "defense"):
        node = nodes_by_key.get(key)
        label = labels[key]
        if node is None:
            lines.append(f"- {label}：暂不参考。当前条件不够，不强行给进攻买点。")
            continue
        lines.append(f"- {label}：{_node_value(node)}，{explanations[key]}")
    return lines


def _compact_price_lines(nodes_by_key: dict[str, CalculatedNode], current_price: float) -> list[str]:
    lines: list[str] = []
    observation = nodes_by_key.get("observation")
    first_buy = nodes_by_key.get("first_buy")
    deep_buy = nodes_by_key.get("deep_buy")
    defense = nodes_by_key.get("defense")
    show_observation = True
    if (
        observation is not None
        and first_buy is not None
        and observation.upper < first_buy.lower
        and current_price > first_buy.upper
    ):
        show_observation = False
    if observation is not None and show_observation:
        lines.append(f"- 观察区：{_node_value(observation)}，先观察是否企稳。")
    if first_buy is not None:
        lines.append(f"- 第一买入区：{_node_value(first_buy)}，企稳后才考虑分批。")
    elif deep_buy is not None:
        lines.append(f"- 深度买入区：{_node_value(deep_buy)}，只用于更深回调。")
    if defense is not None:
        lines.append(f"- 防守位：{_node_value(defense)}，未来买入后的风控预案，不是买点。")
    if not lines:
        lines.append("- 当前没有足够数据生成关键价格。")
    return lines


def _plain_overlap_instruction(nodes_by_key: dict[str, CalculatedNode]) -> str:
    observation = nodes_by_key.get("observation")
    first_buy = nodes_by_key.get("first_buy")
    if observation is None or first_buy is None:
        return _plain_failure_instruction(nodes_by_key)
    overlap_lower = max(observation.lower, first_buy.lower)
    overlap_upper = min(observation.upper, first_buy.upper)
    if overlap_lower <= overlap_upper:
        return (
            f"观察区和第一买入区有重叠（{overlap_lower:.2f}-{overlap_upper:.2f}），"
            "这只是重点观察位置，仍要等企稳确认，不是跌到就自动买。"
        )
    return _plain_failure_instruction(nodes_by_key)


def _one_line_summary(nodes_by_key: dict[str, CalculatedNode], current_price: float) -> str:
    first_buy = nodes_by_key.get("first_buy")
    defense = nodes_by_key.get("defense")
    if first_buy is not None:
        if current_price > first_buy.upper:
            summary = f"现在不追高，先等回到 {first_buy.upper:.2f} 附近或更低，并出现企稳确认后再考虑分批。"
        elif first_buy.lower <= current_price <= first_buy.upper:
            summary = "已经到第一买入区附近，但仍要等企稳确认，不能只看价格自动买。"
        else:
            summary = "价格已经低于第一买入区，先看能否止跌收回，不能急着接下跌。"
    else:
        observation = nodes_by_key.get("observation")
        summary = (
            f"先看观察区 {observation.lower:.2f}-{observation.upper:.2f} 是否企稳，再决定是否有计划。"
            if observation is not None
            else "数据不足时不做买入计划。"
        )
    if defense is not None:
        summary += f" 防守位 {defense.lower:.2f} 只作为未来买入后的风控预案。"
    return summary


def _plain_reason_lines(
    *,
    plan,
    reasons: list[str],
    fundamentals: FundamentalSnapshot | None,
    market_context: MarketContext | None,
    limit: int = 4,
) -> list[str]:
    output: list[str] = []
    if plan.snapshot.ma200 is not None:
        if plan.snapshot.current_price < plan.snapshot.ma200:
            output.append("价格还在长期趋势线 MA200 下方，系统认为趋势偏弱。")
        else:
            output.append("价格在长期趋势线 MA200 上方，趋势没有完全走坏。")
    if plan.snapshot.rsi14 is not None and plan.snapshot.rsi14 >= 70:
        output.append("RSI 偏高，说明短线可能偏热，不适合追高。")
    if fundamentals is not None and fundamentals.next_earnings is not None:
        days = fundamentals.days_to_earnings()
        if days is not None and 0 <= days <= 7:
            output.append(f"距离财报只有 {days} 天，财报前波动可能变大，所以降低行动等级。")
    if market_context is not None and market_context.risk_off:
        output.append("大盘或板块环境偏弱，所以买点要更保守。")
    for reason in reasons:
        if len(output) >= limit:
            break
        if not reason or reason in output:
            continue
        if _skip_plain_reason(reason, output):
            continue
        if reason:
            output.append(reason)
    return [f"- {item}" for item in output[:limit]] or ["- 当前没有足够强的正向信号，所以先等待确认。"]


def _skip_plain_reason(reason: str, existing: list[str]) -> bool:
    if any(fragment in reason for fragment in ("加分", "正向", "未触发", "占优", "校准置信度")):
        return True
    if "RSI" in reason and any("RSI" in item for item in existing):
        return True
    if "MA200" in reason and any("MA200" in item for item in existing):
        return True
    if "分析师" in reason or "评级" in reason:
        return True
    return False


def _node_value(node: CalculatedNode) -> str:
    return f"{node.lower:.2f}" if node.lower == node.upper else f"{node.lower:.2f}-{node.upper:.2f}"


def _build_structured_trade_plan(
    *,
    display_stock: str,
    facts: dict[str, object],
    action_state: ActionState,
    confidence: Confidence,
    nodes_by_key: dict[str, CalculatedNode],
    risk_adjustments: list[str],
    fundamentals: FundamentalSnapshot | None,
    market_context: MarketContext | None,
    score_breakdown: ScoreBreakdown,
) -> TradePlanPayload:
    snapshot = facts.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("trade plan facts did not include snapshot")
    return TradePlanPayload(
        ticker=str(facts.get("ticker") or snapshot.get("ticker") or display_stock),
        display_stock=display_stock,
        current_price=float(snapshot["current_price"]),
        as_of=str(snapshot.get("as_of") or ""),
        action_state=action_state,
        confidence=confidence,
        risk_state=str(facts.get("risk_state") or ""),
        note=str(facts.get("note") or ""),
        trend_stage=str(facts.get("trend_stage") or ""),
        metrics=TradePlanMetrics(
            ma20=_float_or_none(snapshot.get("ma20")),
            ma50=_float_or_none(snapshot.get("ma50")),
            ma200=_float_or_none(snapshot.get("ma200")),
            rsi14=_float_or_none(snapshot.get("rsi14")),
            atr14=_float_or_none(snapshot.get("atr14")),
            high_60=_float_or_none(snapshot.get("high_60")),
            low_60=_float_or_none(snapshot.get("low_60")),
            high_120=_float_or_none(snapshot.get("high_120")),
            low_120=_float_or_none(snapshot.get("low_120")),
            high_252=_float_or_none(snapshot.get("high_252")),
            low_252=_float_or_none(snapshot.get("low_252")),
            weak_state=bool(snapshot.get("weak_state")),
            data_points=int(snapshot.get("data_points") or 0),
        ),
        volume=TradePlanVolume(**_dict_or_empty(facts.get("volume_profile"))),
        backtest=TradePlanBacktest(**_dict_or_empty(facts.get("backtest_calibration"))),
        score_breakdown=TradePlanScoreBreakdown(**score_breakdown.as_facts()),
        parameter_source=str(facts.get("parameter_source") or "default_formula"),
        calibration=TradePlanCalibration(**_dict_or_empty(facts.get("calibration"))),
        backtest_summary=TradePlanBacktestSummary(**_dict_or_empty(facts.get("backtest_summary"))),
        nodes=_structured_nodes(nodes_by_key),
        confirmation_condition=_confirmation_condition(action_state),
        failure_condition=_failure_condition(nodes_by_key),
        risk_adjustments=list(risk_adjustments),
        fundamentals=fundamentals.as_facts() if fundamentals else None,
        market_context=market_context.as_facts() if market_context else None,
    )


def _structured_nodes(nodes_by_key: dict[str, CalculatedNode]) -> list[TradePlanNode]:
    specs = [
        ("observation", "观察区", "MA50 to MA50 - 1 * ATR14"),
        ("first_buy", "第一买入区", "60日高点回撤区间"),
        ("deep_buy", "深度买入区", "120日高点回撤区间，且不高于 MA200 上方 8%"),
        ("defense", "防守位", "min(MA200, current price - ATR multiple * ATR14)"),
    ]
    result: list[TradePlanNode] = []
    for key, title, inactive_formula in specs:
        node = nodes_by_key.get(key)
        if node is None:
            result.append(
                TradePlanNode(
                    key=key,
                    title=title,
                    active=False,
                    action="当前条件未激活，等待确认。",
                    formula=inactive_formula,
                    role_label=_node_role_label(key),
                    plain_explanation=_inactive_node_explanation(key),
                )
            )
            continue
        result.append(
            TradePlanNode(
                key=node.key,
                title=title,
                active=True,
                lower=round(node.lower, 2),
                upper=round(node.upper, 2),
                action=node.action,
                formula=node.formula,
                role_label=_node_role_label(node.key),
                plain_explanation=_node_plain_explanation(node),
            )
        )
    return result


def _node_role_label(key: str) -> str:
    return {
        "observation": "观察参考",
        "first_buy": "分批候选",
        "deep_buy": "深回调候选",
        "defense": "买入后风控",
    }.get(key, "参考")


def _inactive_node_explanation(key: str) -> str:
    if key == "defense":
        return "当前没有可执行买入计划，所以防守位暂时只是预案。"
    return "当前条件未激活，等待价格和量能确认。"


def _node_plain_explanation(node: CalculatedNode) -> str:
    if node.key == "defense":
        return (
            f"{_node_value(node)} 是未来买入后的风控预案，不是买入点；"
            "如果未来已经按计划买入，有效跌破后停止加仓或降低风险。"
        )
    if node.key == "first_buy":
        return f"{_node_value(node)} 是分批候选区，只有回撤压力减弱并企稳后才考虑。"
    if node.key == "deep_buy":
        return f"{_node_value(node)} 只用于更深回调后的候选计划，不能用于追高。"
    if node.key == "observation":
        return f"{_node_value(node)} 是观察区，只看是否企稳，不是自动买入信号。"
    return node.action


def _classify_plan(
    *,
    plan,
    fundamentals: FundamentalSnapshot | None,
    score_breakdown: ScoreBreakdown,
) -> tuple[ActionState, Confidence, list[str]]:
    score = score_breakdown.total
    reasons: list[str] = list(score_breakdown.reasons)
    earnings_window = False
    if fundamentals is not None:
        days = fundamentals.days_to_earnings()
        earnings_window = days is not None and 0 <= days <= 7
    if plan.snapshot.weak_state or plan.trend_stage == "weak_below_ma200" or earnings_window:
        action_state: ActionState = "wait"
    elif plan.trend_stage == "extended_overheated":
        action_state = "wait"
    elif score >= 5:
        action_state = "add_allowed"
    elif score >= 2:
        action_state = "starter_allowed"
    else:
        action_state = "wait"
    if not reasons:
        reasons.append("没有触发主要加减分项，仍需等待价格确认。")
    confidence: Confidence = "high" if score >= 5 else "medium" if score >= 2 else "low"
    if action_state == "wait" and confidence == "high":
        confidence = "medium"
    return action_state, confidence, reasons


def _node_lines(nodes_by_key: dict[str, CalculatedNode]) -> list[str]:
    lines: list[str] = []
    for key in ("observation", "first_buy", "deep_buy", "defense"):
        node = nodes_by_key.get(key)
        if node is None:
            lines.append(f"- {_node_title(key)}：未激活。")
            continue
        value = f"{node.lower:.2f}" if node.lower == node.upper else f"{node.lower:.2f}-{node.upper:.2f}"
        lines.append(f"- {_node_title(key)}：{value}；公式：{node.formula}；动作：{node.action}")
    return lines


def _confirmation_condition(action_state: ActionState) -> str:
    if action_state == "wait":
        return "等待价格重新站稳关键均线，且量能不出现放量下跌后再评估。"
    if action_state == "risk_reduce":
        return "只有重新站回 MA50 且防守位收复后，才考虑恢复风险。"
    return "价格进入节点区间后，需要缩量企稳或放量站回 MA50 才允许分批执行。"


def _failure_condition(nodes_by_key: dict[str, CalculatedNode]) -> str:
    defense = nodes_by_key.get("defense")
    if defense:
        return (
            f"如果未来已经按计划买入，有效跌破 {defense.lower:.2f} 后停止加仓或降低风险，"
            "等待新的交易结构。"
        )
    observation = nodes_by_key.get("observation")
    if observation:
        return f"跌破观察区下沿 {observation.lower:.2f} 且无法快速收回时失效。"
    return "关键数据不足时不生成防守位，不能执行交易计划。"


def _fundamental_lines(fundamentals: FundamentalSnapshot | None) -> list[str]:
    if fundamentals is None:
        return ["- 基本面数据不可用，基本面分不做正向加分。"]
    lines: list[str] = []
    if fundamentals.next_earnings:
        days = fundamentals.days_to_earnings()
        suffix = f"，距今 {days} 天" if days is not None else ""
        lines.append(f"- 下一次财报：{fundamentals.next_earnings.date}{suffix}。")
    if fundamentals.latest_earnings:
        lines.append(f"- 最新 EPS surprise：{_fmt(fundamentals.latest_earnings.surprise_percent)}%。")
    if fundamentals.latest_recommendation:
        item = fundamentals.latest_recommendation
        lines.append(
            f"- 分析师评级：strongBuy={item.strong_buy}，buy={item.buy}，hold={item.hold}，"
            f"sell+strongSell={item.sell + item.strong_sell}。"
        )
    return lines


def _market_lines(market_context: MarketContext | None) -> list[str]:
    if market_context is None:
        return ["- 大盘/板块数据不可用，市场环境分不做正向加分。"]
    return [
        f"- {item.ticker}：{item.trend}，当前价 {item.current_price:.2f}，MA200={_fmt(item.ma200)}。"
        for item in market_context.indices
    ]


def _insufficient_result(display_stock: str, *, reason: str) -> TradePlanResult:
    answer = "\n".join(
        [
            f"{display_stock} 交易计划数据不足。",
            reason,
            "不会在数据不足时编造买入节点。",
        ]
    )
    return TradePlanResult(
        answer=answer,
        action_state="wait",
        confidence="low",
        facts={"error": reason},
        scenarios=_trade_plan_scenarios(display_stock, "wait", "low"),
    )


def _trade_plan_scenarios(stock: str, action_state: ActionState, confidence: Confidence) -> list[ScenarioCard]:
    return [
        ScenarioCard(
            key="bullish",
            title="上行情景",
            stance=f"{stock} 若站稳关键均线且量能确认，可按“{_action_label(action_state)}”的计划观察。",
            reasoning=f"当前规则评分对应计划等级“{_confidence_label(confidence)}”，节点只作为分批计划参考。",
            risk="若高位延伸或财报窗口临近，不能追高。",
        ),
        ScenarioCard(
            key="neutral",
            title="震荡情景",
            stance="价格在观察区附近反复时，优先等待确认。",
            reasoning="节点需要结合量能和均线收复，不单独用价格触发。",
            risk="震荡阶段容易假突破，仓位应分批。",
        ),
        ScenarioCard(
            key="bearish",
            title="下行情景",
            stance="跌破防守位后停止加仓，等待新结构。",
            reasoning="防守位用于控制计划失效，不用于预测底部。",
            risk="若大盘或板块跌破 MA200，进攻型买点应降级。",
        ),
    ]


def _node_title(key: str) -> str:
    return {
        "observation": "观察区",
        "first_buy": "第一买入区",
        "deep_buy": "深度买入区",
        "defense": "防守位",
    }.get(key, key)


def _action_label(value: ActionState) -> str:
    return {
        "wait": "等待",
        "starter_allowed": "允许试探仓",
        "add_allowed": "允许加仓",
        "risk_reduce": "降低风险",
    }[value]


def _confidence_label(value: Confidence) -> str:
    return {"low": "观察级", "medium": "计划级", "high": "执行级"}[value]


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict_or_empty(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
