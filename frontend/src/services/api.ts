import type {
  ApplicationInfo,
  Conversation,
  ConversationAttachment,
  ConversationMessage,
  ConversationSearchHit,
  ConversationStreamEvent,
  FeedbackPayload,
  Health,
  MemoryRecord,
  MemoryStatus,
  QueueSnapshot,
  SystemInfo,
  TaskRecord,
  ToolInfo,
  ToolCandidate,
  ToolKnowledge,
  TurnResult,
} from "../types";

const API_BASE = import.meta.env.VITE_ATLAS_API_URL ?? "http://127.0.0.1:8000/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Atlas API request failed" }));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  system: () => request<SystemInfo>("/system"),
  tools: () => request<{ tools: ToolInfo[] }>("/tools"),
  toolKnowledge: (query = "") => request<{ tools: ToolKnowledge[] }>(`/tool-knowledge${query ? `?query=${encodeURIComponent(query)}` : ""}`),
  discoverTools: (query: string) => request<{ candidates: ToolCandidate[] }>(`/tool-discovery?query=${encodeURIComponent(query)}`),
  applications: () => request<{ applications: ApplicationInfo[] }>("/applications"),
  tasks: () => request<{ tasks: TaskRecord[] }>("/tasks"),
  queue: () => request<QueueSnapshot>("/queue"),
  task: (id: string) => request<TaskRecord>(`/tasks/${id}`),
  createTask: (requestText: string) =>
    request<TaskRecord>("/tasks", {
      method: "POST",
      body: JSON.stringify({ request: requestText }),
    }),
  approveTask: (id: string) => request<TaskRecord>(`/tasks/${id}/approve`, { method: "POST" }),
  denyTask: (id: string) => request<TaskRecord>(`/tasks/${id}/deny`, { method: "POST" }),
  submitFeedback: (id: string, payload: FeedbackPayload) =>
    request<TaskRecord>(`/tasks/${id}/feedback`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  experience: () => request<ExperienceStatus>("/experience"),

  // -- conversations -------------------------------------------------------
  conversations: () => request<{ conversations: Conversation[] }>("/conversations"),
  createConversation: (title?: string) =>
    request<Conversation>("/conversations", {
      method: "POST",
      body: JSON.stringify(title ? { title } : {}),
    }),
  conversationMessages: (id: string) =>
    request<{ messages: ConversationMessage[] }>(`/conversations/${id}/messages`),
  searchConversations: (query: string, limit = 20) =>
    request<{ results: ConversationSearchHit[] }>(
      `/conversations/search?query=${encodeURIComponent(query)}&limit=${limit}`,
    ),
  renameConversation: (id: string, title: string) =>
    request<Conversation>(`/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteConversation: (id: string) =>
    request<void>(`/conversations/${id}`, { method: "DELETE" }),
  sendMessage: (message: string, conversationId?: string | null, attachments: ConversationAttachment[] = []) =>
    request<TurnResult>("/messages", {
      method: "POST",
      body: JSON.stringify({
        message,
        conversation_id: conversationId ?? null,
        attachments,
      }),
    }),
  cancelTurn: (turnId: string) =>
    request<{ turn_id: string; cancelled: boolean }>(`/turns/${turnId}/cancel`, { method: "POST" }),

  // -- long-term memory ----------------------------------------------------
  memories: (query = "") =>
    request<{ memories: MemoryRecord[]; enabled: boolean; count: number }>(
      `/memory${query ? `?query=${encodeURIComponent(query)}` : ""}`,
    ),
  createMemory: (text: string, kind: string) =>
    request<MemoryRecord>("/memory", { method: "POST", body: JSON.stringify({ text, kind }) }),
  updateMemory: (id: string, text: string) =>
    request<MemoryRecord>(`/memory/${id}`, { method: "PATCH", body: JSON.stringify({ text }) }),
  deleteMemory: (id: string) => request<void>(`/memory/${id}`, { method: "DELETE" }),
  clearMemory: () => request<{ removed: number; count: number }>("/memory/clear", { method: "POST" }),
  setMemoryEnabled: (enabled: boolean) =>
    request<MemoryStatus>("/memory/enabled", {
      method: "POST",
      body: JSON.stringify({ enabled }),
    }),
};

/**
 * Stream one conversational turn as real Server-Sent Events.
 *
 * The events carry what actually happened: real model tokens, real tool
 * invocations, and the terminal execution state. If the connection drops the
 * caller still has the final result persisted on the server, so nothing is
 * silently lost.
 */
export function streamMessage(
  message: string,
  options: {
    conversationId?: string | null;
    attachments?: ConversationAttachment[];
    onEvent: (event: ConversationStreamEvent) => void;
    signal?: AbortSignal;
  },
): Promise<void> {
  return fetch(`${API_BASE}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      conversation_id: options.conversationId ?? null,
      attachments: options.attachments ?? [],
      stream: true,
    }),
    signal: options.signal,
  }).then(async (response) => {
    if (!response.ok || !response.body) {
      const body = await response.json().catch(() => ({ detail: "Atlas streaming request failed" }));
      throw new Error(body.detail ?? `Stream failed (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let eventName = "";
    // Parse SSE frames incrementally so a token is shown as soon as it arrives
    // rather than after the whole turn completes.
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          eventName = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          const raw = line.slice(6);
          let data: Record<string, unknown> = {};
          try {
            data = JSON.parse(raw) as Record<string, unknown>;
          } catch {
            data = { text: raw };
          }
          options.onEvent({ type: eventName || "message", data });
          eventName = "";
        }
      }
    }
  });
}

export interface ExperienceStatus {
  enabled: boolean;
  counts?: { total: number; success: number; failure: number; reliable: number; unevaluated: number };
  failure_categories?: string[];
  failure_category_labels?: Record<string, string>;
  analysis?: {
    available: boolean;
    reason?: string;
    evaluated?: number;
    successes?: number;
    failures?: number;
    corrections?: number;
    proposals?: Array<{ kind: string; evidence: number; statement: string; procedure?: string[] }>;
    /** Always false: proposals are never applied automatically. */
    applied?: boolean;
  };
}
