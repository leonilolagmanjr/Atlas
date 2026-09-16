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

export interface TaskRecord {
  id: string;
  request: string;
  status: TaskStatus;
  response: string | null;
  created_at: number;
  updated_at: number;
  task_id: string | null;
  plan: TaskPlan | null;
  tool_calls: Array<Record<string, unknown>>;
  errors: string[];
  warnings: string[];
}
