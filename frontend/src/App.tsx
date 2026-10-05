import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import {
  BrainCircuit,
  ChevronRight,
  CircleAlert,
  Cpu,
  Database,
  HardDrive,
  Layers3,
  MemoryStick,
  Menu,
  MessageSquare,
  MonitorCog,
  Plus,
  Server,
  X,
  Zap,
} from "lucide-react";
import { api } from "./services/api";
import { ChatWorkspace, MemoryView } from "./ChatWorkspace";
import type {
  Health,
  SystemInfo,
  TaskRecord,
  TaskStatus,
  ToolInfo,
} from "./types";
import "./styles.css";

type Section = "chat" | "tasks" | "memory" | "system";

interface NavItem {
  id: Section;
  label: string;
  icon: LucideIcon;
}

const navItems: NavItem[] = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "tasks", label: "Task history", icon: Layers3 },
  { id: "memory", label: "Memory", icon: Database },
  { id: "system", label: "System", icon: MonitorCog },
];

// Task-history badge threshold: the count is real (from the task API), but it
// is only shown once there is something worth showing.
const TASK_BADGE_ID: Section = "tasks";

// The task API returns the full in-memory history. Only the most recent records
// are rendered, so a long history does not make the view unusable.
const TASK_LIST_LIMIT = 200;

const statusLabels: Record<TaskStatus, string> = {
  PENDING: "Queued",
  RUNNING: "Executing",
  COMPLETED: "Completed",
  FAILED: "Failed",
  CANCELLED: "Cancelled",
  UNCERTAIN: "Uncertain",
  WAITING_FOR_CONFIRMATION: "Approval needed",
};

function App() {
  const [section, setSection] = useState<Section>("chat");
  const [health, setHealth] = useState<Health | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [activeTask, setActiveTask] = useState<TaskRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    const found = api.tasks().then((result) => setTasks(result.tasks)).catch(() => undefined);
    void found;
  }, []);

  // Poll the task list while a task is queued or executing, so the history and
  // the running indicator reflect the backend's real state rather than a
  // one-time snapshot.
  useEffect(() => {
    const inFlight = tasks.some((task) => task.status === "PENDING" || task.status === "RUNNING");
    if (!inFlight) return;
    const timer = window.setInterval(() => {
      api.tasks()
        .then((result) => setTasks(result.tasks))
        .catch((reason: Error) => setError(reason.message));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [tasks]);

  // Keep the health indicator honest: it reflects the last real backend
  // response and is re-checked periodically, instead of being assumed online.
  useEffect(() => {
    let cancelled = false;
    const check = () => {
      api.health()
        .then((next) => {
          if (!cancelled) setHealth(next);
        })
        .catch(() => {
          if (!cancelled) setHealth((current) => (current ? { ...current, status: "offline" } : null));
        });
    };
    check();
    const timer = window.setInterval(check, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const changeSection = (nextSection: Section) => {
    setSection(nextSection);
    setMobileNavOpen(false);
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark"><span> A </span></div>
          <div>
            <strong>ATLAS</strong>
            <small>LOCAL OPERATING LAYER</small>
          </div>
        </div>
        <button className="new-task" onClick={() => changeSection("chat")}><Plus size={16} /> New chat</button>
        <div className="nav-label">Workspace</div>
        <nav>
          {navItems.map(({ id, label, icon: Icon }) => (
            <button key={id} className={`nav-item ${section === id ? "nav-item-active" : ""}`} onClick={() => changeSection(id)}>
              <Icon size={17} strokeWidth={1.8} /> <span>{label}</span>{id === TASK_BADGE_ID && tasks.length > 0 ? <em>{tasks.length > 99 ? "99+" : tasks.length}</em> : null}
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="local-badge">
          <span className={`pulse-dot ${health?.status === "online" ? "" : "pulse-dot-idle"}`} />
          Local runtime <span className={`status-mini ${health?.status === "online" ? "" : "status-mini-offline"}`}>{health?.status === "online" ? "online" : health ? "offline" : "checking"}</span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileNavOpen((open) => !open)} title="Open navigation"><Menu size={20} /></button>
          <div className="breadcrumb"><span>Atlas</span><ChevronRight size={14} /><strong>{navItems.find((item) => item.id === section)?.label ?? "Command"}</strong></div>
            <span className="topbar-status">
            <span className="connection"><span className={`pulse-dot ${health?.status === "online" ? "" : "pulse-dot-idle"}`} /> {health ? (health.status === "online" ? "Local / online" : "Local / offline") : "Connecting"}</span>
            <span className="model-chip"><BrainCircuit size={14} /> {health?.model ?? "Qwen local model"}</span>
            <span className="avatar">LL</span>
          </span>
        </header>

        {error ? <div className="error-strip"><CircleAlert size={15} /> {error}<button onClick={() => setError(null)} aria-label="Dismiss error"><X size={14} /></button></div> : null}

        <div className="content-wrap">
          {section === "chat" ? <ChatWorkspace modelLabel={health?.model ?? "Qwen local model"} /> : null}
          {section === "tasks" ? <TasksView tasks={tasks} activeTask={activeTask} onSelect={setActiveTask} /> : null}
          {section === "system" ? <SystemView system={system} health={health} tools={tools} /> : null}
          {section === "memory" ? <MemoryView /> : null}
        </div>
      </main>
    </div>
  );
}


function StatusPill({ status }: { status: TaskStatus }) {
  return <span className={`status-pill status-${status.toLowerCase()}`}><span />{statusLabels[status]}</span>;
}

function TasksView({ tasks, activeTask, onSelect }: { tasks: TaskRecord[]; activeTask: TaskRecord | null; onSelect: (task: TaskRecord) => void }) {
  const visible = tasks.slice(0, TASK_LIST_LIMIT);
  return <PageFrame eyebrow="Operations / task history" title="Task history" description="Every request stays inspectable through its real execution state."><div className="task-table">{visible.length ? <>{visible.map((task) => <button className={`task-row ${activeTask?.id === task.id ? "task-row-active" : ""}`} key={task.id} onClick={() => onSelect(task)}><span className="row-status"><StatusPill status={task.status} /></span><span className="row-request">{task.request}</span><span className="row-time">{new Date(task.created_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span><ChevronRight size={16} /></button>)}{tasks.length > visible.length ? <p className="chat-empty-note">Showing the {visible.length} most recent of {tasks.length} tasks.</p> : null}</> : <UnavailableMessage title="No tasks yet" detail="Start a conversation in Chat and any work Atlas performs will appear here." />}</div></PageFrame>;
}

function SystemView({ system, health, tools }: { system: SystemInfo | null; health: Health | null; tools: ToolInfo[] }) {
  const disk = system?.system.disk;
  const stats = system?.system;
  return (
    <PageFrame eyebrow="Runtime / local machine" title="System state" description="Observed values from the Atlas computer runtime. Unavailable values are never fabricated.">
      <div className="system-grid">
        <div className="system-card system-card-wide">
          <span className="card-label">Operating system</span>
          <strong>{stats ? `${stats.os} ${stats.release}` : "Unavailable"}</strong>
          <span>{stats?.machine ?? "No machine data"}</span>
        </div>
        <div className="system-card">
          <Cpu size={17} />
          <span className="card-label">CPU</span>
          <strong>{stats?.cpu_count ? `${stats.cpu_count} cores` : "Unavailable"}</strong>
        </div>
        <div className="system-card">
          <Server size={17} />
          <span className="card-label">Model</span>
          <strong>{health?.model ?? "Unavailable"}</strong>
        </div>
        <div className="system-card system-card-wide">
          <HardDrive size={17} />
          <span className="card-label">Disk</span>
          <strong>{disk ? `${formatBytes(disk.free)} free of ${formatBytes(disk.total)}` : "Unavailable"}</strong>
          <div className="disk-bar">
            <span style={{ width: disk && disk.total ? `${Math.round((disk.used / disk.total) * 100)}%` : "0%" }} />
          </div>
        </div>
        <div className="system-card">
          <MemoryStick size={17} />
          <span className="card-label">Memory</span>
          <strong>Unavailable</strong>
          <span>Metric not exposed yet</span>
        </div>
        <div className="system-card">
          <Cpu size={17} />
          <span className="card-label">Processor</span>
          <strong>{systemStat(stats?.processor ?? null)}</strong>
          <span>{stats?.python ? `Python ${stats.python}` : "Runtime unavailable"}</span>
        </div>
      </div>
      <div className="tool-section-label">Registered runtime tools ({tools.length})</div>
      <div className="tool-grid">
        {tools.length ? (
          tools.map((tool) => (
            <div className="tool-card" key={tool.name}>
              <div className="tool-card-top">
                <span className="tool-symbol"><Zap size={16} /></span>
                <span className={`risk risk-${tool.risk_level}`}>{tool.permission_level.replaceAll("_", " ")}</span>
              </div>
              <strong>{tool.name}</strong>
              <p>{tool.description}</p>
              <small>{tool.category}</small>
            </div>
          ))
        ) : (
          <UnavailableMessage title="Tool registry unavailable" detail="Start the Atlas API to inspect real capabilities." />
        )}
      </div>
    </PageFrame>
  );
}

function systemStat(value: string | null | undefined, fallback = "Unavailable") {
  return value && value.trim() ? value : fallback;
}

function PageFrame({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children: ReactNode }) {
  return <section className="page-frame"><div className="eyebrow"><span className="eyebrow-line" /> {eyebrow}</div><h1>{title}</h1><p className="lead page-lead">{description}</p>{children}</section>;
}

function UnavailableMessage({ title, detail }: { title: string; detail: string }) {
  return <div className="unavailable"><Database size={20} /><strong>{title}</strong><span>{detail}</span></div>;
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

export default App;
