import { createMockSession, streamMockChat } from "./mock.js";

const runtimeConfig = globalThis.__INVEST_DH_CONFIG__ || {};
const API_BASE =
  runtimeConfig.API_BASE_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8020";
const FORCE_MOCK =
  String(runtimeConfig.USE_MOCK ?? import.meta.env.VITE_USE_MOCK ?? "").toLowerCase() === "true";

export async function createSession() {
  if (FORCE_MOCK) {
    return createMockSession();
  }
  const response = await fetch(`${API_BASE}/api/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    throw new Error(`创建会话失败：${response.status}`);
  }
  return await response.json();
}

export async function listSessions() {
  if (FORCE_MOCK) {
    return [];
  }
  const response = await fetch(`${API_BASE}/api/sessions`);
  if (!response.ok) {
    throw new Error(`获取最近对话失败：${response.status}`);
  }
  return await response.json();
}

export async function getSession(sessionId) {
  if (FORCE_MOCK) {
    return { session_id: sessionId, title: "Mock 对话", messages: [] };
  }
  const response = await fetch(`${API_BASE}/api/session/${encodeURIComponent(sessionId)}`);
  if (!response.ok) {
    throw new Error(`获取对话历史失败：${response.status}`);
  }
  return await response.json();
}

export async function streamChat({ sessionId, message, onEvent }) {
  if (FORCE_MOCK) {
    return streamMockChat({ onEvent, sessionId, message });
  }
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
    }),
  });
  if (!response.ok || !response.body) {
    throw new Error(`请求聊天接口失败：${response.status}`);
  }
  await parseSse(response.body, onEvent);
}

async function parseSse(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const event = parseEvent(part);
      if (event) {
        onEvent(event.type, event.data);
      }
    }
  }

  const finalEvent = parseEvent(buffer);
  if (finalEvent) {
    onEvent(finalEvent.type, finalEvent.data);
  }
}

function parseEvent(block) {
  const lines = block.split("\n");
  let type = "message";
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      type = line.replace("event:", "").trim();
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.replace("data:", "").trim());
    }
  }
  if (!dataLines.length) {
    return null;
  }
  return {
    type,
    data: JSON.parse(dataLines.join("\n")),
  };
}
