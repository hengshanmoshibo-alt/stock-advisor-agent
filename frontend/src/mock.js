const STOCK_FIXTURES = {
  MSFT: {
    ticker: "MSFT",
    display_stock: "微软",
    current_price: 415.75,
    firstBand: [351.26, 383.25],
    deepBand: [327.84, 351.26],
    observationBand: [382.81, 393.65],
    defense: 382.81,
    ma20: 405.31,
    ma50: 393.65,
    ma200: 382.81,
    rsi14: 47.8,
    atr14: 10.84,
    high60: 468.35,
    high120: 468.35,
    trend_stage: "pullback_above_ma200",
    parameter_source: "calibrated",
    firstDrawdown: [0.18, 0.25],
    hitRate60: 1,
    avgReturn60: 0.118,
    avgDrawdown120: -0.071,
    stopLossRate120: 0.125,
    triggerCount: 8,
    volumeNote: "缩量回撤，说明抛压暂时没有明显放大。",
  },
  NVDA: {
    ticker: "NVDA",
    display_stock: "英伟达",
    current_price: 875.42,
    firstBand: [728.36, 805.03],
    deepBand: [651.69, 728.36],
    observationBand: [812.2, 842.6],
    defense: 801.9,
    ma20: 861.35,
    ma50: 842.6,
    ma200: 801.9,
    rsi14: 53.4,
    atr14: 30.4,
    high60: 958.37,
    high120: 958.37,
    trend_stage: "pullback_above_ma200",
    parameter_source: "calibrated",
    firstDrawdown: [0.16, 0.24],
    hitRate60: 0.875,
    avgReturn60: 0.164,
    avgDrawdown120: -0.118,
    stopLossRate120: 0.167,
    triggerCount: 12,
    volumeNote: "回撤量能温和，暂未出现连续放量下跌。",
  },
  AMD: {
    ticker: "AMD",
    display_stock: "AMD",
    current_price: 148.62,
    firstBand: [128.1, 139.82],
    deepBand: [113.45, 128.1],
    observationBand: [137.7, 143.55],
    defense: 132.25,
    ma20: 146.8,
    ma50: 143.55,
    ma200: 132.25,
    rsi14: 49.6,
    atr14: 5.85,
    high60: 159.23,
    high120: 162.07,
    trend_stage: "pullback_above_ma200",
    parameter_source: "calibrated",
    firstDrawdown: [0.12, 0.18],
    hitRate60: 0.692,
    avgReturn60: 0.097,
    avgDrawdown120: -0.142,
    stopLossRate120: 0.231,
    triggerCount: 13,
    volumeNote: "波动偏大，买点需要更重视分批和防守位。",
  },
  UNH: {
    ticker: "UNH",
    display_stock: "联合健康",
    current_price: 505.2,
    firstBand: [442.8, 471.6],
    deepBand: [398.5, 433.9],
    observationBand: [472.4, 489.6],
    defense: 468.7,
    ma20: 498.3,
    ma50: 489.6,
    ma200: 468.7,
    rsi14: 46.2,
    atr14: 17.2,
    high60: 589.5,
    high120: 619.8,
    trend_stage: "pullback_above_ma200",
    parameter_source: "mock_formula",
    firstDrawdown: [0.2, 0.25],
    hitRate60: 0.714,
    avgReturn60: 0.084,
    avgDrawdown120: -0.096,
    stopLossRate120: 0.143,
    triggerCount: 7,
    volumeNote: "医疗保健股波动通常低于高贝塔科技股，回撤区间更重视趋势防守。",
  },
};

export async function createMockSession() {
  await wait(120);
  return { session_id: `mock-${Date.now()}` };
}

export async function streamMockChat({ onEvent, sessionId = "mock-session", message = "", error }) {
  const fixture = pickFixture(message);
  const structured = buildStructuredPayload(fixture, sessionId, error);

  onEvent("meta", {
    request_id: `mock-${Date.now()}`,
    session_id: sessionId,
    model_mode: "mock",
  });

  for (const chunk of chunkText(structured.answer, 22)) {
    await wait(30);
    onEvent("delta", { text: chunk });
  }

  await wait(100);
  onEvent("structured", structured);
  onEvent("done", { request_id: `mock-done-${Date.now()}` });
}

function pickFixture(message) {
  const text = String(message || "").toLowerCase();
  if (text.includes("微软") || text.includes("msft") || text.includes("microsoft")) {
    return STOCK_FIXTURES.MSFT;
  }
  if (text.includes("英伟达") || text.includes("nvda") || text.includes("nvidia")) {
    return STOCK_FIXTURES.NVDA;
  }
  if (text.includes("amd") || text.includes("超威")) {
    return STOCK_FIXTURES.AMD;
  }
  if (text.includes("联合健康") || text.includes("联合保健") || text.includes("unh") || text.includes("unitedhealth")) {
    return STOCK_FIXTURES.UNH;
  }
  return null;
}

function buildStructuredPayload(fixture, sessionId, error) {
  if (!fixture) {
    return buildUnknownPayload(sessionId, error);
  }

  const tradePlan = buildTradePlan(fixture);
  const answer = buildAnswer(fixture, error);

  return {
    answer,
    answer_mode: "trade_plan_agent",
    trade_plan: tradePlan,
    scenarios: buildScenarios(fixture),
    citations: [],
    disclaimer: "以上内容由 AI 和确定性工具生成，仅供研究参考，不构成投资建议。",
    session_state: {
      session_id: sessionId,
      turn_count: 1,
      summary: "",
      model_mode: "mock",
    },
  };
}

function buildUnknownPayload(sessionId, error) {
  const prefix = error ? "当前后端不可用。" : "";
  return {
    answer: `${prefix}mock 演示数据没有识别到这只股票，因此不生成交易计划卡片，避免把其他股票错误显示成微软。请启动真实后端后再查询，或在 mock 模式下输入：微软、英伟达、AMD、联合健康。`,
    answer_mode: "mock_unavailable",
    trade_plan: null,
    scenarios: [],
    citations: [],
    disclaimer: "以上内容为前端 mock 演示提示，不构成投资建议。",
    session_state: {
      session_id: sessionId,
      turn_count: 1,
      summary: "",
      model_mode: "mock",
    },
  };
}

function buildTradePlan(fixture) {
  return {
    ticker: fixture.ticker,
    display_stock: fixture.display_stock,
    current_price: fixture.current_price,
    as_of: "2026-04-24",
    action_state: "wait",
    confidence: "medium",
    risk_state: "normal",
    note: "Mock 演示：价格和节点为前端预览数据，真实结果以后台行情和回测计算为准。",
    trend_stage: fixture.trend_stage,
    parameter_source: fixture.parameter_source,
    metrics: {
      ma20: fixture.ma20,
      ma50: fixture.ma50,
      ma200: fixture.ma200,
      rsi14: fixture.rsi14,
      atr14: fixture.atr14,
      high_60: fixture.high60,
      low_60: fixture.firstBand[0],
      high_120: fixture.high120,
      low_120: fixture.deepBand[0],
      data_points: 501,
      weak_state: false,
    },
    volume: {
      average_volume_20: 24500000,
      average_volume_50: 27100000,
      volume_ratio_20: 0.76,
      volume_ratio_50: 0.69,
      signal: "quiet_pullback",
      note: fixture.volumeNote,
    },
    calibration: {
      source_key: fixture.ticker.toLowerCase(),
      generated_at: "2026-04-25T00:42:09+00:00",
      parameter_source: fixture.parameter_source,
      confidence_adjustment: 1,
      usable: true,
      parameters: {
        first_buy_drawdown_band: fixture.firstDrawdown,
        deep_buy_drawdown_band: [0.25, 0.3],
        defense_atr_multiplier: 2,
        min_sample_count: 8,
      },
    },
    backtest_summary: {
      data_points: 501,
      parameter_source: fixture.parameter_source,
      parameters: {
        first_buy_drawdown_band: fixture.firstDrawdown,
        deep_buy_drawdown_band: [0.25, 0.3],
        defense_atr_multiplier: 2,
        min_sample_count: 8,
      },
      node_stats: {
        first_buy: {
          node_key: "first_buy",
          trigger_count: fixture.triggerCount,
          hit_rate_20d: Math.max(0, fixture.hitRate60 - 0.2),
          hit_rate_60d: fixture.hitRate60,
          hit_rate_120d: Math.min(1, fixture.hitRate60 + 0.08),
          average_return_20d: fixture.avgReturn60 * 0.45,
          average_return_60d: fixture.avgReturn60,
          average_return_120d: fixture.avgReturn60 * 1.45,
          average_max_drawdown_120d: fixture.avgDrawdown120,
          stop_loss_rate_120d: fixture.stopLossRate120,
        },
      },
    },
    score_breakdown: {
      technical: 1,
      volume: 1,
      fundamentals: 1,
      market: -2,
      event_risk: 0,
      total: 1,
      reasons: ["缩量回撤给量能加分", "QQQ 或 SPY 处于风险状态", "校准置信度修正：+1"],
    },
    nodes: [
      {
        key: "observation",
        title: "观察区",
        active: true,
        lower: fixture.observationBand[0],
        upper: fixture.observationBand[1],
        action: "等待价格在该区域企稳。",
        formula: "MA50 到 MA50 - 1 * ATR14",
      },
      {
        key: "first_buy",
        title: "第一买入区",
        active: true,
        lower: fixture.firstBand[0],
        upper: fixture.firstBand[1],
        action: "只在回撤减速后分批试探。",
        formula: `60 日高点回撤 ${formatBand(fixture.firstDrawdown)}；历史校准`,
      },
      {
        key: "deep_buy",
        title: "深度买入区",
        active: true,
        lower: fixture.deepBand[0],
        upper: fixture.deepBand[1],
        action: "深度回撤才考虑，不能追高。",
        formula: "120 日高点回撤 25%-30%，且受 MA200 * 1.08 约束",
      },
      {
        key: "defense",
        title: "防守位",
        active: true,
        lower: fixture.defense,
        upper: fixture.defense,
        action: "有效跌破后停止加仓。",
        formula: "min(MA200, current price - 2 * ATR14)",
      },
    ],
    confirmation_condition: "价格进入节点区间后，需要缩量企稳或放量站回 MA50，才允许分批执行。",
    failure_condition: `有效跌破 ${fixture.defense.toFixed(2)} 后停止加仓，等待新结构。`,
    risk_adjustments: ["大盘环境偏弱，进攻型买点降级。", "校准样本可用，但仍需等待价格确认。"],
    fundamentals: null,
    market_context: {
      indices: [
        { ticker: "QQQ", current_price: 510.12, ma50: 514.2, ma200: 489.3, trend: "above_ma200" },
        { ticker: "SMH", current_price: 238.4, ma50: 241.2, ma200: 220.1, trend: "above_ma200" },
      ],
      risk_off: false,
    },
  };
}

function buildAnswer(fixture, error) {
  const fallbackNote = error ? "当前后端不可用，以下为 mock 演示数据。" : "以下为 mock 演示数据。";
  return `${fallbackNote}${fixture.display_stock} 当前交易计划偏向等待确认。第一买入区是 ${formatRange(
    fixture.firstBand,
  )}，深度买入区是 ${formatRange(fixture.deepBand)}，防守位是 ${fixture.defense.toFixed(
    2,
  )}。当前价 ${fixture.current_price.toFixed(2)} 仍高于主要买点区间，适合先观察，不建议追高。`;
}

function buildScenarios(fixture) {
  return [
    {
      key: "bullish",
      title: "上行情景",
      stance: "重新站回 MA50 后，可考虑试探仓。",
      reasoning: "价格结构未转弱，但当前未进入高性价比区间。",
      risk: "若大盘转弱，进攻型买点需要降级。",
    },
    {
      key: "neutral",
      title: "震荡情景",
      stance: `继续等待价格接近 ${formatRange(fixture.observationBand)}。`,
      reasoning: "缩量回撤有利，但确认信号还不够。",
      risk: "横盘阶段容易假突破。",
    },
    {
      key: "bearish",
      title: "下行情景",
      stance: `跌破 ${fixture.defense.toFixed(2)} 后停止加仓。`,
      reasoning: "防守位用于控制计划失效。",
      risk: "若跌破 MA200，需要重新评估趋势阶段。",
    },
  ];
}

function formatRange(range) {
  return `${range[0].toFixed(2)} - ${range[1].toFixed(2)}`;
}

function formatBand(range) {
  return `${Math.round(range[0] * 100)}%-${Math.round(range[1] * 100)}%`;
}

function chunkText(text, size) {
  const chunks = [];
  for (let index = 0; index < text.length; index += size) {
    chunks.push(text.slice(index, index + size));
  }
  return chunks;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
