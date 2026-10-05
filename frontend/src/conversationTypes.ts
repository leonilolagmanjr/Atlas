
export interface ToolCall {
  tool?: string;
  status?: string;
  success?: boolean;
  parameters?: Record<string, unknown>;
  output?: unknown;
  error?: string | null;
}

/**
 * Conversations (persistent, first-class)
 */

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  last_modified: string;
  message_count: number;
  summary?: string | null;
  archived: boolean;
  model?: string;
  metadata?: Record<string, unknown>;
}

/** How a turn was served. Reported from what actually happened. */
export type ResponseKind =
  | "conversation"
  | "retrieval"
  | "research"
  | "task"
  | "computer"
  | "hybrid"
  | "clarification"
  | "memory"
  | "system"
  | "error";

/** The truthful recorded state of a turn. */
export type ExecutionState =
  | "completed"
  | "failed"
  | "cancelled"
  | "waiting_for_confirmation"
  | "running"
  | "unknown";

export interface ConversationAttachment {
  name: string;
  kind?: string;
  path?: string;
  size?: number;
  /** Set when the file could not be read; shown instead of pretending it was. */
  error?: string;
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  turn_number: number;
  attachments: ConversationAttachment[];
  tool_calls: ToolCall[];
  tool_results: Array<{ tool?: string; output?: unknown; error?: string | null }>;
  citations: string[];
  execution_state: ExecutionState;
  metadata: {
    response_kind?: ResponseKind | "";
    activity?: string[];
    used_model?: boolean;
    task_id?: string | null;
    [key: string]: unknown;
  };
}

export interface ConversationSearchHit {
  conversation_id: string;
  title: string;
  message_id: string;
  role: string;
  excerpt: string;
  timestamp: string | null;
  score: number;
  match: "semantic" | "message" | "summary";
}

export interface TurnResult {
  conversation_id: string;
  message_id: string;
  user_message_id: string;
  text: string;
  kind: ResponseKind;
  execution_state: ExecutionState;
  tool_calls: ToolCall[];
  tool_results: Array<{ tool?: string; output?: unknown; error?: string | null }>;
  citations: string[];
  activity: string[];
  classification?: {
    kind: string;
    delegate: boolean;
    reason: string;
    signals: string[];
    context_dependent: boolean;
    anaphoric: boolean;
  };
  metadata: {
    memory_written?: Array<{ id: string; text: string; kind: string; summary?: string }>;
    [key: string]: unknown;
  };
}

/** One real Server-Sent Event from a streaming turn. */
export interface ConversationStreamEvent {
  type: string;
  data: Record<string, unknown>;
}

/** The result of approving or denying a confirmation-paused conversational turn. */
export interface ConfirmationResolution {
  conversation_id: string;
  task_id: string;
  execution_state: ExecutionState;
  message: ConversationMessage | null;
  messages: ConversationMessage[];
}

// ---------------------------------------------------------------------------
// Long-term memory
// ---------------------------------------------------------------------------

export type MemoryKind = "preference" | "instruction" | "fact" | "project" | "workflow";

export interface MemoryRecord {
  id: string;
  kind: MemoryKind;
  text: string;
  summary: string;
  source: string;
  conversation_id: string;
  message_id: string;
  confidence: number;
  created_at: string;
  updated_at: string;
}

export interface MemoryStatus {
  enabled: boolean;
  count: number;
  kinds: string[];
}

/** The canonical memory kinds, mirrored from the backend vocabulary. */
export const MEMORY_KINDS: Array<{ value: MemoryKind; label: string }> = [
  { value: "preference", label: "Preference" },
  { value: "instruction", label: "Instruction" },
  { value: "fact", label: "Fact" },
  { value: "project", label: "Project" },
  { value: "workflow", label: "Workflow" },
];
