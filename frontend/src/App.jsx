import { startTransition, useEffect, useMemo, useState } from "react";
import { createSession, getSession, listSessions, streamChat } from "./api.js";

const STARTER_PROMPTS = [
  "微软现在买入节点是多少？",
  "英伟达如果跌下来，第一买点和防守位是多少？",
  "AMD 这个买点历史胜率怎么样？",
];

const MODE_LABELS = {
  mock: "Mock 演示",
  fallback: "本地模板",
  "openai-compatible": "外部模型",
  ollama: "Ollama",
  "ollama:trade_plan": "Ollama 交易计划",
  "openai-compatible:trade_plan": "外部模型交易计划",
  failover: "本地优先，外部兜底",
  trade_plan_agent: "规则交易计划",
  "trade_plan_agent:cached": "规则交易计划缓存",
  trade_plan_agent_ai: "AI 润色交易计划",
  trade_plan_context: "连续追问",
  stock_advisor_explainer: "概念解释",
  stock_advisor_clarify: "需要补充信息",
  stock_advisor_llm_router: "Agent 路由",
  trade_plan_insufficient: "数据不足",
  connecting: "连接中",
  offline: "后端未连接",
};

const ACTION_LABELS = {
  wait: "等待",
  starter_allowed: "允许试探仓",
  add_allowed: "允许加仓",
  risk_reduce: "降低风险",
};

const PLAN_LEVEL_LABELS = {
  low: "观察级",
  medium: "计划级",
  high: "执行级",
};

const PLAN_LEVEL_DESCRIPTIONS = {
  low: "当前条件不足，只适合观察，不适合执行买入。",
  medium: "条件部分满足，可以按计划小仓位观察，但仍需要确认信号。",
  high: "条件相对充分，可以按计划执行，但仍要遵守失效条件。",
};

const TREND_LABELS = {
  strong_uptrend: "强趋势",
  normal_uptrend: "正常上升趋势",
  pullback_above_ma200: "MA200 上方回撤",
  weak_below_ma200: "跌破 MA200 弱势",
  extended_overheated: "过热延伸",
  range_or_unclear: "震荡或不清晰",
};

const PARAMETER_SOURCE_LABELS = {
  calibrated: "历史校准",
  default_formula: "默认公式",
};

export default function App() {
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [statusText, setStatusText] = useState("正在初始化会话...");
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelMode, setModelMode] = useState("connecting");
  const [recentConversations, setRecentConversations] = useState([]);

  useEffect(() => {
    let active = true;
    createSession()
      .then((payload) => {
        if (!active) return;
        setSessionId(payload.session_id);
        setStatusText("会话已就绪，可以输入股票或市场问题。");
        refreshSessions(payload.session_id);
      })
      .catch(() => {
        if (!active) return;
        setSessionId("");
        setModelMode("offline");
        setStatusText("后端不可用。请先启动股票建议后端，再刷新页面。");
      });
    return () => {
      active = false;
    };
  }, []);

  const conversationItems = useMemo(() => {
    return recentConversations
      .map((item) => ({ ...item, active: item.session_id === sessionId }))
      .slice(0, 8);
  }, [recentConversations, sessionId]);

  async function refreshSessions(activeSessionId = sessionId) {
    try {
      const sessions = await listSessions();
      setRecentConversations(sessions);
      if (activeSessionId) setSessionId(activeSessionId);
    } catch {
      // 会话列表不是主链路，失败不影响聊天。
    }
  }

  async function handleNewChat() {
    try {
      const payload = await createSession();
      setSessionId(payload.session_id);
      setMessages([]);
      setStatusText("新对话已创建。");
      refreshSessions(payload.session_id);
    } catch {
      setStatusText("新建对话失败，请确认后端已启动。");
    }
  }

  async function handleSelectConversation(selectedSessionId) {
    if (!selectedSessionId || selectedSessionId === sessionId || isStreaming) return;
    try {
      const payload = await getSession(selectedSessionId);
      setSessionId(payload.session_id);
      setMessages(payload.messages || []);
      setStatusText("已恢复历史对话。");
      refreshSessions(payload.session_id);
    } catch {
      setStatusText("恢复历史对话失败，可能后端已重启。");
      refreshSessions();
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || !sessionId || isStreaming) return;

    const assistantId = `assistant-${Date.now()}`;
    setInput("");
    setIsStreaming(true);
    setStatusText("正在连接 Agent...");
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", text: trimmed },
      {
        id: assistantId,
        role: "assistant",
        text: "",
        progressText: "正在连接 Agent...",
        progressEvents: [],
        scenarios: [],
        citations: [],
        disclaimer: "",
        session_state: null,
        trade_plan: null,
      },
    ]);

    try {
      await streamChat({
        sessionId,
        message: trimmed,
        onEvent(type, payload) {
          if (type === "meta") {
            setModelMode(payload.model_mode || "connecting");
            updateAssistantProgress(assistantId, "已连接后端，正在启动 Multi-Agent。");
            return;
          }
          if (type === "progress") {
            const label = payload.label || "正在处理";
            setStatusText(label);
            appendAssistantProgress(assistantId, payload);
            return;
          }
          if (type === "delta") {
            startTransition(() => {
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? { ...message, text: `${message.text}${payload.text}`, progressText: "" }
                    : message,
                ),
              );
            });
            return;
          }
          if (type === "structured") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId ? { ...message, ...payload, progressText: "" } : message,
              ),
            );
            setStatusText("回答完成。");
            return;
          }
          if (type === "error") {
            setStatusText(payload.message || "请求失败，请稍后重试。");
            updateAssistantProgress(assistantId, "");
            setIsStreaming(false);
            return;
          }
          if (type === "done") {
            setStatusText("就绪。");
            updateAssistantProgress(assistantId, "");
            setIsStreaming(false);
            refreshSessions(sessionId);
          }
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "请求失败，请稍后重试。";
      setStatusText(message);
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId
            ? { ...item, text: `后端请求失败：${message}`, progressText: "" }
            : item,
        ),
      );
      setIsStreaming(false);
    }
  }

  function updateAssistantProgress(assistantId, progressText) {
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId && !message.text ? { ...message, progressText } : message,
      ),
    );
  }

  function appendAssistantProgress(assistantId, progress) {
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              progressText: progress.label,
              progressEvents: [...(message.progressEvents || []), progress],
            }
          : message,
      ),
    );
  }

  return (
    <div className="page">
      <aside className="sidebar">
        <div className="sidebar-top">
          <button type="button" className="new-chat-button" onClick={handleNewChat}>
            <span>+</span>
            <span>新建对话</span>
          </button>
        </div>

        <div className="sidebar-section recent">
          <p className="sidebar-label">最近对话</p>
          <div className="item-list">
            {conversationItems.map((item) => (
              <SidebarRow
                key={item.session_id}
                label={item.title}
                muted={!item.active}
                active={item.active}
                onClick={() => handleSelectConversation(item.session_id)}
              />
            ))}
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-left">
            <strong>智能投顾助手</strong>
            <span>{MODE_LABELS[modelMode] || modelMode}</span>
          </div>
        </header>

        <section className={`conversation ${messages.length ? "thread-mode" : "home-mode"}`}>
          {messages.length === 0 ? (
            <div className="hero">
              <p className="eyebrow">Multi-Agent Trade Plan</p>
              <h1>把买入节点变成可验证的交易计划</h1>
              <p>{statusText}</p>
            </div>
          ) : (
            <div className="thread">
              {messages.map((message) => (
                <MessageBlock key={message.id} message={message} />
              ))}
            </div>
          )}
        </section>

        <footer className="composer-shell">
          <div className="composer">
            {!messages.length ? (
              <div className="starter-row">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className="starter-chip"
                    onClick={() => setInput(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            ) : null}

            <form className="composer-form" onSubmit={handleSubmit}>
              <button type="button" className="composer-leading" aria-label="添加">
                +
              </button>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                rows={1}
                placeholder="输入你想分析的股票，例如：MSFT、NVDA、AMD..."
              />
              <div className="composer-actions">
                <button
                  type="submit"
                  className="submit-button"
                  disabled={!sessionId || isStreaming || !input.trim()}
                  aria-label="发送"
                >
                  {isStreaming ? "…" : "→"}
                </button>
              </div>
            </form>

            <p className="composer-note">
              内容由 AI 和确定性工具生成，仅供研究参考，不构成投资建议。{statusText}
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}

function SidebarRow({ label, muted = false, active = false, onClick }) {
  return (
    <button
      type="button"
      className={`sidebar-row ${muted ? "muted" : ""} ${active ? "active" : ""}`}
      onClick={onClick}
    >
      <span>{label}</span>
    </button>
  );
}

function MessageBlock({ message }) {
  const isAssistant = message.role === "assistant";
  return (
    <article className={`message-block ${message.role}`}>
      <div className="message-avatar">{isAssistant ? "AI" : "你"}</div>
      <div className="message-content">
        <div className="message-head">
          <strong>{isAssistant ? "智能投顾助手" : "你"}</strong>
          {isAssistant && message.session_state ? (
            <span>
              {MODE_LABELS[message.session_state.model_mode] || message.session_state.model_mode}
            </span>
          ) : null}
        </div>

        {isAssistant && message.progressEvents?.length && !message.text ? (
          <AgentProgress events={message.progressEvents} />
        ) : null}

        <div className={`message-text ${isAssistant && !message.text ? "pending-text" : ""}`}>
          {message.text || message.progressText || "正在生成回答..."}
        </div>

        {isAssistant && message.trade_plan ? <TradePlanCard tradePlan={message.trade_plan} /> : null}

        {isAssistant && !message.trade_plan && message.scenarios?.length ? (
          <div className="scenario-grid">
            {message.scenarios.map((scenario) => (
              <section key={scenario.key} className="scenario-panel">
                <span>{scenario.title}</span>
                <h4>{scenario.stance}</h4>
                <p>{scenario.reasoning}</p>
                <small>风险：{scenario.risk}</small>
              </section>
            ))}
          </div>
        ) : null}

        {isAssistant && message.citations?.length ? (
          <div className="evidence-panel">
            <h4>引用依据</h4>
            <div className="evidence-list">
              {message.citations.map((citation) => (
                <details key={`${citation.title}-${citation.published}`} className="evidence-card">
                  <summary>
                    <div>
                      <strong>{citation.title}</strong>
                      <span>{citation.published}</span>
                    </div>
                    <em>相关度 {formatNumber(citation.score)}</em>
                  </summary>
                  <p>{citation.snippet}</p>
                  <a href={citation.url} target="_blank" rel="noreferrer">
                    打开原文
                  </a>
                </details>
              ))}
            </div>
          </div>
        ) : null}

        {isAssistant && message.disclaimer ? (
          <div className="disclaimer">{message.disclaimer}</div>
        ) : null}
      </div>
    </article>
  );
}

function AgentProgress({ events }) {
  return (
    <div className="agent-progress">
      {events.map((event, index) => (
        <span key={`${event.agent || event.step}-${index}`}>
          {event.agent ? <strong>{event.agent}</strong> : null}
          {event.label}
        </span>
      ))}
    </div>
  );
}

function TradePlanCard({ tradePlan }) {
  const firstBuyStats = tradePlan.backtest_summary?.node_stats?.first_buy;
  const score = tradePlan.score_breakdown;

  return (
    <section className="trade-plan-card">
      <div className="trade-plan-header">
        <div>
          <span className="card-kicker">结构化交易计划</span>
          <h3>
            {tradePlan.display_stock || tradePlan.ticker} <span>{tradePlan.ticker}</span>
          </h3>
        </div>
        <div className={`action-pill ${tradePlan.action_state || "wait"}`}>
          {ACTION_LABELS[tradePlan.action_state] || tradePlan.action_state}
        </div>
      </div>

      <section className="plain-summary">
        <h4>{plainActionTitle(tradePlan.action_state)}</h4>
        <p>{plainActionText(tradePlan)}</p>
      </section>

      <div className="metric-strip">
        <Metric label="当前价" value={formatPrice(tradePlan.current_price)} />
        <Metric label="计划等级" value={PLAN_LEVEL_LABELS[tradePlan.confidence] || tradePlan.confidence} />
        <Metric label="趋势阶段" value={TREND_LABELS[tradePlan.trend_stage] || tradePlan.trend_stage || "未知"} />
        <Metric
          label="参数来源"
          value={PARAMETER_SOURCE_LABELS[tradePlan.parameter_source] || tradePlan.parameter_source}
          tone={tradePlan.parameter_source === "calibrated" ? "positive" : "neutral"}
        />
      </div>

      <p className="plan-level-note">
        说明：{PLAN_LEVEL_DESCRIPTIONS[tradePlan.confidence] || "当前条件需要结合价格、趋势和确认信号一起看。"}
      </p>

      <NodeList nodes={tradePlan.nodes || []} currentPrice={tradePlan.current_price} />

      <div className="plan-grid">
        <section className="plan-panel">
          <h4>确认条件</h4>
          <p>{tradePlan.confirmation_condition || "等待价格和量能确认。"}</p>
        </section>
        <section className="plan-panel">
          <h4>失效条件</h4>
          <p>{tradePlan.failure_condition || "关键数据不足，暂不执行计划。"}</p>
        </section>
      </div>

      <details className="plan-details">
        <summary>查看专业细节：回测、评分和风险修正</summary>
        <div className="plan-grid">
          <BacktestSummary stats={firstBuyStats} dataPoints={tradePlan.backtest_summary?.data_points} />
          <ScoreBreakdown score={score} />
        </div>
        {tradePlan.risk_adjustments?.length ? (
          <section className="plan-panel risk-panel">
            <h4>风险修正</h4>
            <ul>
              {tradePlan.risk_adjustments.map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          </section>
        ) : null}
      </details>
    </section>
  );
}

function plainActionTitle(actionState) {
  if (actionState === "starter_allowed") return "结论：可以小仓试探";
  if (actionState === "add_allowed") return "结论：可以按计划加仓";
  if (actionState === "risk_reduce") return "结论：先控制风险";
  return "结论：现在先等";
}

function plainActionText(tradePlan) {
  const firstBuy = tradePlan.nodes?.find((node) => node.key === "first_buy" && node.active);
  const observation = tradePlan.nodes?.find((node) => node.key === "observation" && node.active);
  const defense = tradePlan.nodes?.find((node) => node.key === "defense" && node.active);
  const currentPrice = Number(tradePlan.current_price);
  const inObservation =
    observation &&
    Number.isFinite(currentPrice) &&
    observation.lower != null &&
    observation.upper != null &&
    currentPrice >= Number(observation.lower) &&
    currentPrice <= Number(observation.upper);
  const buyText = firstBuy
    ? `第一买入区在 ${formatRange(firstBuy.lower, firstBuy.upper)}，没到之前不要追高。`
    : observation
      ? inObservation
        ? `当前价已经在观察区 ${formatRange(observation.lower, observation.upper)} 内，意思是先看企稳，不是立刻买。`
        : `当前只给观察区 ${formatRange(observation.lower, observation.upper)}，意思是先看企稳，不是立刻买。`
      : "当前没有可参考的买入区。";
  const defenseText = defense
    ? `防守位 ${formatRange(defense.lower, defense.upper)} 是买入后的风控预案，不是当前买点。`
    : "当前没有激活防守位，按观察区下沿作为失效参考。";
  return `${buyText}${defenseText}`;
}

function NodeList({ nodes, currentPrice }) {
  const orderedKeys = ["observation", "first_buy", "deep_buy", "defense"];
  const ordered = orderedKeys
    .map((key) => nodes.find((node) => node.key === key))
    .filter(Boolean)
    .concat(nodes.filter((node) => !orderedKeys.includes(node.key)));

  return (
    <div className="node-grid">
      {ordered.map((node) => (
        <section key={node.key} className={`node-card ${node.active ? "active" : "inactive"}`}>
          <div className="node-card-head">
            <h4>{node.title || node.key}</h4>
            <span>{nodeRoleLabel(node)}</span>
          </div>
          <strong>{node.active ? formatRange(node.lower, node.upper) : "暂不参考"}</strong>
          <p>{nodeActionText(node, currentPrice)}</p>
          <small>{node.formula}</small>
        </section>
      ))}
    </div>
  );
}

function nodeActionText(node, currentPrice) {
  if (!node?.active) return node?.plain_explanation || node?.action || "当前条件未满足。";
  if (node.plain_explanation) return node.plain_explanation;
  if (node.key !== "observation") return node.action || "当前条件未满足。";
  const price = Number(currentPrice);
  const lower = Number(node.lower);
  const upper = Number(node.upper);
  if (!Number.isFinite(price) || !Number.isFinite(lower) || !Number.isFinite(upper)) {
    return node.action || "当前条件未满足。";
  }
  if (price >= lower && price <= upper) {
    return "已进入观察区；先看能否企稳确认，不是立刻买入信号。";
  }
  if (price > upper) return "等待价格回到观察区并企稳；未进入前不追高。";
  return "价格已跌破观察区，先等待重新收回区间。";
}

function nodeRoleLabel(node) {
  if (node.role_label) return node.role_label;
  if (!node.active) return "暂不参考";
  return {
    observation: "观察参考",
    first_buy: "分批候选",
    deep_buy: "深回调候选",
    defense: "买入后风控",
  }[node.key] || "参考";
}

function BacktestSummary({ stats, dataPoints }) {
  return (
    <section className="plan-panel">
      <h4>回测摘要：第一买入区</h4>
      <div className="fact-list">
        <Metric label="K线样本" value={dataPoints ? `${dataPoints} 条` : "暂无"} />
        <Metric label="触发次数" value={stats?.trigger_count ?? "暂无"} />
        <Metric label="60日胜率" value={formatPercent(stats?.hit_rate_60d)} tone="positive" />
        <Metric label="60日平均收益" value={formatPercent(stats?.average_return_60d)} tone="positive" />
        <Metric label="120日最大回撤" value={formatPercent(stats?.average_max_drawdown_120d)} tone="negative" />
        <Metric label="止损触发率" value={formatPercent(stats?.stop_loss_rate_120d)} tone="negative" />
      </div>
    </section>
  );
}

function ScoreBreakdown({ score }) {
  const rows = [
    ["技术", score?.technical],
    ["量能", score?.volume],
    ["基本面", score?.fundamentals],
    ["市场", score?.market],
    ["事件风险", score?.event_risk],
    ["总分", score?.total],
  ];

  return (
    <section className="plan-panel">
      <h4>评分拆解</h4>
      <div className="score-bars">
        {rows.map(([label, value]) => (
          <div key={label} className="score-row">
            <span>{label}</span>
            <div className="score-track">
              <i
                className={Number(value) >= 0 ? "positive" : "negative"}
                style={{ width: `${Math.min(Math.abs(Number(value) || 0) * 18 + 8, 100)}%` }}
              />
            </div>
            <strong>{value ?? 0}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function Metric({ label, value, tone = "neutral" }) {
  return (
    <div className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value ?? "暂无"}</strong>
    </div>
  );
}

function formatRange(lower, upper) {
  if (lower == null && upper == null) return "暂无";
  if (lower === upper || upper == null) return formatPrice(lower);
  return `${formatPrice(lower)} - ${formatPrice(upper)}`;
}

function formatPrice(value) {
  if (value == null || Number.isNaN(Number(value))) return "暂无";
  return Number(value).toFixed(2);
}

function formatPercent(value) {
  if (value == null || Number.isNaN(Number(value))) return "暂无";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "0.00";
  return Number(value).toFixed(2);
}
