import type {
  ApplicationInfo,
  FeedbackPayload,
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
  submitFeedback: (id: string, payload: FeedbackPayload) =>
    request<TaskRecord>(`/tasks/${id}/feedback`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  experience: () => request<ExperienceStatus>("/experience"),
};

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
