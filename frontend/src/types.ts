export type TaskStatus =
  | "PENDING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED"
  | "UNCERTAIN"
  | "WAITING_FOR_CONFIRMATION";

// Conversation types live in their own module (they are a separate subsystem
// from the task API) and are re-exported here so UI code has one import site.
export type {
  Conversation,
  ConversationAttachment,
  ConversationMessage,
  ConversationSearchHit,
  ConversationStreamEvent,
  ConfirmationResolution,
  ExecutionState,
  MemoryKind,
  MemoryRecord,
  MemoryStatus,
  ResponseKind,
} from "./conversationTypes";

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

export interface WebResult {
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
  web_results: WebResult[];
  reasoning?: ReasoningStep[];
  provenance?: string[];
  citations?: string[];
  response_mode?: string | null;
  evidence?: EvidenceSummary | null;
  feedback_available?: boolean;
  feedback_outcome?: FeedbackOutcome;
  feedback_category?: string;
  feedback_category_label?: string;
  feedback_reason?: string;
  feedback_correction?: string;
  experience_id?: string;
}

/** Task outcome feedback. "unknown" means the user has not answered yet. */
export type FeedbackOutcome = "success" | "failure" | "unknown";

export interface FeedbackPayload {
  outcome: "success" | "failure";
  reason?: string;
  failure_category?: string;
  correction?: string;
  expected_behavior?: string;
  response_quality?: "positive" | "negative" | "";
}
