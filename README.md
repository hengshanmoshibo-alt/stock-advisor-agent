# Stock Advisor Agent

一个本机运行的美股交易计划 Agent。系统通过 LLM 理解用户问题，但价格节点、观察区、买入区、防守位、评分和回测结果全部由后端确定性工具计算，避免模型编造价格。

> 仅用于投资研究和交易计划参考，不构成投资建议。

## 能力概览

- **交易计划**：回答“微软现在能不能买”“英伟达第一买点是多少”“防守位在哪里”等问题。
- **技术节点计算**：基于日 K 计算 MA20/MA50/MA200、RSI14、ATR14、60/120/252 日高低点。
- **多因子评分**：结合趋势、量能、基本面、市场环境和事件风险，输出计划等级。
- **回测校准**：使用历史 K 线校准部分买入区间和置信度修正。
- **多轮上下文**：同一会话内支持“那第一买点呢”“防守位是什么意思”“为什么不是现在买”等追问。
- **概念解释**：支持 MA200、RSI、ATR、防守位等概念说明。
- **前端聊天页**：React + Vite，支持流式回答、结构化交易计划卡片、最近对话恢复。

当前版本聚焦 **美股和美股 ETF** 的交易计划生成。

## 系统边界

LLM 负责：

- 判断用户意图。
- 识别美股名称或 ticker。
- 组织中文回答。
- 解释确定性工具的结果。

## Agent 流程

```text
用户问题
  -> LLM 判断节点
  -> 工具执行
     - quote_lookup
     - market_candles
     - fundamentals
     - market_context
     - technical_nodes
     - backtest_calibration
  -> 确定性交易计划计算
  -> LLM 润色
  -> 后端一致性校验
  -> SSE 输出 answer + structured.trade_plan
```

## 项目结构

```text
frontend/                     React + Vite 前端
scripts/                      本机启动和校准脚本
src/invest_digital_human/     后端 Agent、工具和 API
tests/                        单元测试
data/calibration/             小型回测校准缓存
projects/stock-advisor/       项目入口说明
```

## 本机启动

### 1. 准备环境变量

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填入本机 API key。不要提交 `.env`。

常用变量：

```text
FINNHUB_API_KEY=your_finnhub_key
MASSIVE_API_KEY=your_massive_key
OLLAMA_MODEL=gemma3:4b
API_KEY=your_openai_compatible_key
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat
ENABLE_TRADE_PLAN_LLM=true
```

### 2. 启动后端和前端

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_stock_advisor_local.ps1
```

访问：

```text
http://127.0.0.1:5173/
```

后端健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8020/api/health
```

只启动后端：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_stock_advisor_local.ps1 -NoFrontend
```

## 手动启动

后端：

```powershell
$env:PYTHONPATH='src'
$env:PORT='8020'
python scripts\run_stock_advisor_backend.py
```

前端：

```powershell
cd frontend
npm install
npm run dev
```

## API

创建会话：

```http
POST /api/session
```

聊天 SSE：

```http
POST /api/chat
```

最近对话：

```http
GET /api/sessions
GET /api/session/{session_id}
```

健康检查：

```http
GET /api/health
```

## 测试

后端测试：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover tests
```

前端构建：

```powershell
cd frontend
npm ci
npm run build
```

当前基线：

```text
59 backend tests
frontend build passes
```

## 建议验收问题

```text
MA200 是什么？
买入节点
英伟达现在能不能买？
那第一买点是多少？
防守位呢？
为什么不是现在买？
换成微软呢？
中国银行呢？
```

预期行为：

- 概念问题不显示交易计划卡片。
- 无股票问题要求补充美股名称或 ticker。
- 非美股问题明确说明当前不支持。
- 交易计划回答包含当前价、买入区、确认条件、失效条件和风险提示。
- 防守位始终解释为买入后的风控预案，不是买点。

## 数据与密钥

不会提交：

- `.env`
- API key
- `frontend/node_modules`
- `frontend/dist`
- 日志、截图、临时文件
- 大型文章索引或向量文件

当前仓库只保留运行股票建议 Agent 所需的代码、配置样例、测试和小型校准数据。

## 免责声明

本项目输出只用于投资研究和交易计划参考，不构成投资建议。市场数据、模型识别和第三方接口都可能出错。真实交易前，请结合自己的期限、仓位和风险承受能力独立判断。
