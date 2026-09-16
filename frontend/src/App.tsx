import { useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  AppWindow,
  ArrowUpRight,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleAlert,
  Clock3,
  Command,
  Cpu,
  Database,
  FileSearch,
  FolderOpen,
  HardDrive,
  History,
  Layers3,
  LockKeyhole,
  MemoryStick,
  Menu,
  MonitorCog,
  Play,
  Plus,
  RotateCw,
  Search,
  Send,
  Server,
  Settings2,
  ShieldCheck,
  SquareTerminal,
  X,
  Zap,
} from "lucide-react";
import { api } from "./services/api";
import type {
  ApplicationInfo,
  Health,
  SystemInfo,
  TaskRecord,
  TaskStatus,
  ToolInfo,
} from "./types";
import "./styles.css";

type Section = "command" | "tasks" | "files" | "knowledge" | "applications" | "tools" | "memory" | "system" | "settings";

interface NavItem {
  id: Section;
  label: string;
  icon: LucideIcon;
  live?: boolean;
}

const navItems: NavItem[] = [
  { id: "command", label: "Command", icon: Command },
  { id: "tasks", label: "Tasks", icon: Layers3, live: true },
  { id: "files", label: "Files", icon: FolderOpen },
  { id: "knowledge", label: "Knowledge", icon: FileSearch },
  { id: "applications", label: "Applications", icon: AppWindow },
  { id: "tools", label: "Tools", icon: SquareTerminal },
  { id: "memory", label: "Memory", icon: Database },
  { id: "system", label: "System", icon: MonitorCog },
];

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
  const [section, setSection] = useState<Section>("command");
  const [health, setHealth] = useState<Health | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [applications, setApplications] = useState<ApplicationInfo[]>([]);
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [activeTask, setActiveTask] = useState<TaskRecord | null>(null);
  const [request, setRequest] = useState("");
  const [loading, setLoading] = useState(false);
  const [appsLoading, setAppsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const refreshTasks = () => api.tasks().then((result) => setTasks(result.tasks)).catch(() => undefined);

  useEffect(() => {
    Promise.all([api.health(), api.system(), api.tools(), api.tasks()])
      .then(([nextHealth, nextSystem, nextTools, nextTasks]) => {
        setHealth(nextHealth);
        setSystem(nextSystem);
        setTools(nextTools.tools);
        setTasks(nextTasks.tasks);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (section !== "applications" || applications.length > 0 || appsLoading) return;
    setAppsLoading(true);
    api.applications()
      .then((result) => setApplications(result.applications))
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setAppsLoading(false));
  }, [section, applications.length, appsLoading]);

  useEffect(() => {
    if (!activeTask || !["PENDING", "RUNNING"].includes(activeTask.status)) return;
    const timer = window.setInterval(() => {
      api.task(activeTask.id)
        .then((nextTask) => {
          setActiveTask(nextTask);
          setTasks((current) => [nextTask, ...current.filter((task) => task.id !== nextTask.id)]);
        })
        .catch((reason: Error) => setError(reason.message));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [activeTask]);

  const completedCount = useMemo(
    () => tasks.filter((task) => task.status === "COMPLETED").length,
    [tasks],
  );

  async function submitTask(event: React.FormEvent) {
    event.preventDefault();
    const cleanRequest = request.trim();
    if (!cleanRequest || loading) return;
    setLoading(true);
    setError(null);
    try {
      const task = await api.createTask(cleanRequest);
      setActiveTask(task);
      setTasks((current) => [task, ...current.filter((item) => item.id !== task.id)]);
      setRequest("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not submit task");
    } finally {
      setLoading(false);
    }
  }

  async function approveTask() {
    if (!activeTask) return;
    try {
      const task = await api.approveTask(activeTask.id);
      setActiveTask(task);
      refreshTasks();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Approval failed");
    }
  }

  async function denyTask() {
    if (!activeTask) return;
    try {
      const task = await api.denyTask(activeTask.id);
      setActiveTask(task);
      refreshTasks();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Cancellation failed");
    }
  }

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
        <button className="new-task" onClick={() => changeSection("command")}><Plus size={16} /> New task</button>
        <div className="nav-label">Workspace</div>
        <nav>
          {navItems.map(({ id, label, icon: Icon, live }) => (
            <button key={id} className={`nav-item ${section === id ? "nav-item-active" : ""}`} onClick={() => changeSection(id)}>
              <Icon size={17} strokeWidth={1.8} /> <span>{label}</span>{live && tasks.length > 0 ? <em>{tasks.length}</em> : null}
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="local-badge"><span className="pulse-dot" /> Local runtime <span className="status-mini">online</span></div>
        <button className={`nav-item ${section === "settings" ? "nav-item-active" : ""}`} onClick={() => changeSection("settings")}><Settings2 size={17} /> <span>Settings</span></button>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileNavOpen((open) => !open)} title="Open navigation"><Menu size={20} /></button>
          <div className="breadcrumb"><span>Atlas</span><ChevronRight size={14} /><strong>{navItems.find((item) => item.id === section)?.label ?? "Command"}</strong></div>
          <div className="topbar-status">
            <span className="connection"><span className="pulse-dot" /> {health?.status === "online" ? "Local / online" : "Connecting"}</span>
            <span className="model-chip"><BrainCircuit size={14} /> {health?.model ?? "Qwen local model"}</span>
            <span className="avatar">LL</span>
          </div>
        </header>

        {error ? <div className="error-strip"><CircleAlert size={15} /> {error}<button onClick={() => setError(null)}><X size={14} /></button></div> : null}

        <div className="content-wrap">
          {section === "command" ? <CommandWorkspace activeTask={activeTask} request={request} setRequest={setRequest} loading={loading} onSubmit={submitTask} onApprove={approveTask} onDeny={denyTask} completedCount={completedCount} /> : null}
          {section === "tasks" ? <TasksView tasks={tasks} activeTask={activeTask} onSelect={setActiveTask} /> : null}
          {section === "applications" ? <ApplicationsView applications={applications} loading={appsLoading} /> : null}
          {section === "tools" ? <ToolsView tools={tools} /> : null}
          {section === "system" ? <SystemView system={system} health={health} /> : null}
          {section === "files" || section === "knowledge" || section === "memory" || section === "settings" ? <UnavailableView section={section} /> : null}
        </div>
      </main>
    </div>
  );
}

function CommandWorkspace({
  activeTask,
  request,
  setRequest,
  loading,
  onSubmit,
  onApprove,
  onDeny,
  completedCount,
}: {
  activeTask: TaskRecord | null;
  request: string;
  setRequest: (value: string) => void;
  loading: boolean;
  onSubmit: (event: FormEvent) => void;
  onApprove: () => void;
  onDeny: () => void;
  completedCount: number;
}) {
  return (
    <div className="workspace-grid">
      <section className="command-column">
        <div className="eyebrow"><span className="eyebrow-line" /> Command interface</div>
        <h1>What should Atlas<br /><span>take care of?</span></h1>
        <p className="lead">Describe a task in plain language. Atlas will plan it, request approval when needed, and report only verified execution state.</p>
        <form className="command-box" onSubmit={onSubmit}>
          <div className="command-box-top"><span className="input-prompt">›</span><textarea value={request} onChange={(event) => setRequest(event.target.value)} placeholder="Ask Atlas to inspect, search, or act..." rows={3} /></div>
          <div className="command-box-bottom"><span className="hint"><kbd>Enter</kbd> submit · <kbd>Shift Enter</kbd> new line</span><button className="send-button" disabled={loading || !request.trim()} title="Submit task">{loading ? <RotateCw size={17} className="spin" /> : <Send size={17} />}</button></div>
        </form>
        <div className="suggestion-row">
          <button onClick={() => setRequest("Inspect my system and report the current disk space")}>Inspect system <ArrowUpRight size={13} /></button>
          <button onClick={() => setRequest("Find all PDF files in the project")}>Find project PDFs <ArrowUpRight size={13} /></button>
          <button onClick={() => setRequest("Search my knowledge base for the latest project status")}>Search knowledge <ArrowUpRight size={13} /></button>
        </div>
        {activeTask ? <TaskPanel task={activeTask} onApprove={onApprove} onDeny={onDeny} /> : <EmptyTaskState completedCount={completedCount} />}
      </section>
      <aside className="right-rail">
        <div className="rail-header"><span>Runtime telemetry</span><Activity size={15} /></div>
        <Metric label="CPU cores" value="Available" icon={Cpu} />
        <Metric label="Memory" value="Local process" icon={MemoryStick} />
        <Metric label="Storage" value="Indexed locally" icon={HardDrive} />
        <div className="rail-divider" />
        <div className="rail-note"><ShieldCheck size={16} /><div><strong>Permission-aware</strong><span>Consequential actions pause for your approval.</span></div></div>
        <div className="rail-note"><LockKeyhole size={16} /><div><strong>Local first</strong><span>Reasoning and data stay on this machine.</span></div></div>
      </aside>
    </div>
  );
}

function Metric({ label, value, icon: Icon }: { label: string; value: string; icon: LucideIcon }) {
  return <div className="metric"><Icon size={16} /><div><span>{label}</span><strong>{value}</strong></div></div>;
}

function EmptyTaskState({ completedCount }: { completedCount: number }) {
  return <div className="empty-task"><Clock3 size={17} /><span>{completedCount ? `${completedCount} task${completedCount === 1 ? "" : "s"} completed this session.` : "No active task. Atlas is ready."}</span></div>;
}

function TaskPanel({ task, onApprove, onDeny }: { task: TaskRecord; onApprove: () => void; onDeny: () => void }) {
  const waiting = task.status === "WAITING_FOR_CONFIRMATION";
  return (
    <section className="task-panel">
      <div className="section-heading"><div><span className="eyebrow">Current task</span><h2>{statusLabels[task.status]}</h2></div><StatusPill status={task.status} /></div>
      <p className="task-request">{task.request}</p>
      {waiting ? <div className="approval-box"><div><LockKeyhole size={18} /><div><strong>Atlas is ready to act</strong><span>This action changes your computer. Review the plan and approve it to continue.</span></div></div><div className="approval-actions"><button className="button-muted" onClick={onDeny}><X size={15} /> Deny</button><button className="button-primary" onClick={onApprove}><Check size={15} /> Approve action</button></div></div> : null}
      {task.plan ? <div className="plan-list"><div className="plan-label">Execution plan</div>{task.plan.steps.map((step) => <div className="plan-step" key={step.id}><span className={`step-icon ${step.status.toLowerCase()}`}><StepIcon status={step.status} /></span><span>{step.name}</span><small>{step.status.replaceAll("_", " ").toLowerCase()}</small></div>)}</div> : null}
      {task.response ? <div className="result-box"><span>Atlas result</span><p>{task.response}</p></div> : null}
      {task.errors.length ? <div className="failure-box"><CircleAlert size={15} /> {task.errors.join(" ")}</div> : null}
    </section>
  );
}

function StepIcon({ status }: { status: string }) {
  if (status === "COMPLETED") return <Check size={13} />;
  if (status === "FAILED") return <X size={13} />;
  if (status === "RUNNING") return <Play size={11} fill="currentColor" />;
  return <span className="step-dot" />;
}

function StatusPill({ status }: { status: TaskStatus }) {
  return <span className={`status-pill status-${status.toLowerCase()}`}><span />{statusLabels[status]}</span>;
}

function TasksView({ tasks, activeTask, onSelect }: { tasks: TaskRecord[]; activeTask: TaskRecord | null; onSelect: (task: TaskRecord) => void }) {
  return <PageFrame eyebrow="Operations / task history" title="Task history" description="Every request stays inspectable through its real execution state."><div className="task-table">{tasks.length ? tasks.map((task) => <button className={`task-row ${activeTask?.id === task.id ? "task-row-active" : ""}`} key={task.id} onClick={() => onSelect(task)}><span className="row-status"><StatusPill status={task.status} /></span><span className="row-request">{task.request}</span><span className="row-time">{new Date(task.created_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span><ChevronRight size={16} /></button>) : <UnavailableMessage title="No tasks yet" detail="Submit a command from the Command workspace to create a real task." />}</div></PageFrame>;
}

function ApplicationsView({ applications, loading }: { applications: ApplicationInfo[]; loading: boolean }) {
  return <PageFrame eyebrow="Computer / detected software" title="Applications" description="Installed applications detected from Windows uninstall metadata."><div className="data-grid">{loading ? <LoadingState /> : applications.length ? applications.map((application) => <div className="application-row" key={`${application.name}-${application.version}`}><div className="app-icon"><AppWindow size={18} /></div><div><strong>{application.name}</strong><span>{application.publisher || "Publisher unavailable"}</span></div><small>{application.version || "Version unavailable"}</small><span className="availability"><span /> Detected</span></div>) : <UnavailableMessage title="No application data" detail="The backend returned no installed application entries." />}</div></PageFrame>;
}

function ToolsView({ tools }: { tools: ToolInfo[] }) {
  return <PageFrame eyebrow="Runtime / capability registry" title="Available tools" description="Capabilities currently exposed by the Atlas backend and their permission boundaries."><div className="tool-grid">{tools.map((tool) => <div className="tool-card" key={tool.name}><div className="tool-card-top"><span className="tool-symbol"><Zap size={16} /></span><span className={`risk risk-${tool.risk_level}`}>{tool.permission_level.replaceAll("_", " ")}</span></div><strong>{tool.name}</strong><p>{tool.description}</p><small>{tool.category}</small></div>)}{!tools.length ? <UnavailableMessage title="Tool registry unavailable" detail="Start the Atlas API to inspect real capabilities." /> : null}</div></PageFrame>;
}

function SystemView({ system, health }: { system: SystemInfo | null; health: Health | null }) {
  const disk = system?.system.disk;
  return <PageFrame eyebrow="Runtime / local machine" title="System state" description="Observed values from the Atlas computer runtime. Unavailable values are never fabricated."><div className="system-grid"><div className="system-card system-card-wide"><span className="card-label">Operating system</span><strong>{system ? `${system.system.os} ${system.system.release}` : "Unavailable"}</strong><span>{system?.system.machine ?? "No machine data"}</span></div><div className="system-card"><Cpu size={17} /><span className="card-label">CPU</span><strong>{system?.system.cpu_count ? `${system.system.cpu_count} cores` : "Unavailable"}</strong></div><div className="system-card"><Server size={17} /><span className="card-label">Model</span><strong>{health?.model ?? "Unavailable"}</strong></div><div className="system-card system-card-wide"><HardDrive size={17} /><span className="card-label">Disk</span><strong>{disk ? `${formatBytes(disk.free)} free of ${formatBytes(disk.total)}` : "Unavailable"}</strong><div className="disk-bar"><span style={{ width: disk ? `${Math.round((disk.used / disk.total) * 100)}%` : "0%" }} /></div></div><div className="system-card"><MemoryStick size={17} /><span className="card-label">Memory</span><strong>Unavailable</strong><span>Metric not exposed yet</span></div><div className="system-card"><Database size={17} /><span className="card-label">Vector database</span><strong>Local</strong><span>Chroma runtime</span></div></div></PageFrame>;
}

function UnavailableView({ section }: { section: Section }) {
  const labels: Record<string, [string, string]> = { files: ["Files", "File management endpoints are not exposed by the backend yet."], knowledge: ["Knowledge base", "Indexing and retrieval diagnostics need a dedicated API surface."], memory: ["Memory", "Session memory exists in the backend; management endpoints are next."], settings: ["Settings", "Runtime configuration is file-based today. Editable settings are not exposed yet."] };
  const [title, detail] = labels[section];
  return <PageFrame eyebrow="Backend-dependent surface" title={title} description={detail}><UnavailableMessage title="Integration pending" detail="This surface is intentionally not pretending to work. The existing CLI/runtime remains the source of truth." /></PageFrame>;
}

function PageFrame({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children: ReactNode }) {
  return <section className="page-frame"><div className="eyebrow"><span className="eyebrow-line" /> {eyebrow}</div><h1>{title}</h1><p className="lead page-lead">{description}</p>{children}</section>;
}

function UnavailableMessage({ title, detail }: { title: string; detail: string }) {
  return <div className="unavailable"><Database size={20} /><strong>{title}</strong><span>{detail}</span></div>;
}

function LoadingState() {
  return <div className="loading-state"><RotateCw size={18} className="spin" /> Reading local runtime...</div>;
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

export default App;
