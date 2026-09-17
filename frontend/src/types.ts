export type TaskStatus =
  | "PENDING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED"
  | "UNCERTAIN"
  | "WAITING_FOR_CONFIRMATION";

export interface Health {
  status: string;
  model: string;
  local: boolean;
  api: string;
}

export interface ToolInfo {
  name: string;
  description: string;
  category: string;
  permission_level: string;
  risk_level: string;
}

export interface ToolKnowledge {
  name: string;
  type: string;
  category: string;
  description: string;
  purpose: string[];
  examples: string[];
  aliases: string[];
  related_tools: string[];
  requires_admin: boolean;
  risk_level: string;
  destructive: boolean;
  read_only: boolean;
  expected_output: string;
  platforms: string[];
  powershell_versions: string[];
  metadata: Record<string, unknown>;
}

export interface ToolCandidate {
  tool: string;
  tool_type: string;
  reason: string;
  risk_level: string;
  read_only: boolean;
  score: number;
}

export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
  thumbnail_url?: string | null;
}

export interface SystemInfo {
  system: {
    os: string;
    release: string;
    version: string;
    machine: string;
    processor: string;
    python: string;
    cpu_count: number | null;
    disk: { path: string; total: number; used: number; free: number };
  };
}

export interface ApplicationInfo {
  name: string;
  version: string;
  publisher: string;
  install_location: string;
}

export interface PlanStep {
  id: string;
  name: string;
  action: string;
  description: string;
  status: string;
  metadata: Record<string, unknown>;
}

export interface TaskPlan {
  id: string;
  status: string;
  steps: PlanStep[];
}

export interface ToolCall {
  tool?: string;
  status?: string;
  success?: boolean;
  parameters?: Record<string, unknown>;
  output?: unknown;
  error?: string | null;
}

export interface QueueSnapshot {
  running: string | null;
  pending: string[];
}

export interface ReasoningStep {
  stage: "UNDERSTANDING" | "SOURCE_SELECTION" | "RETRIEVING" | "PLANNING" | "EXECUTING" | "VERIFYING" | "EVALUATING" | "ANSWERING";
  detail: string;
  iteration: number;
}

export interface EvidenceSummary {
  count: number;
  sources: string[];
}

export interface TaskRecord {
  id: string;
  request: string;
  status: TaskStatus;
  response: string | null;
  created_at: number;
  updated_at: number;
  task_id: string | null;
  plan: TaskPlan | null;
  tool_calls: ToolCall[];
  errors: string[];
  warnings: string[];
  web_sources: string[];
  reasoning?: ReasoningStep[];
  provenance?: string[];
  citations?: string[];
  response_mode?: string | null;
  evidence?: EvidenceSummary | null;
}
