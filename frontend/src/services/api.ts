import type {
  ApplicationInfo,
  Health,
  QueueSnapshot,
  SystemInfo,
  TaskRecord,
  ToolInfo,
  ToolCandidate,
  ToolKnowledge,
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
};
