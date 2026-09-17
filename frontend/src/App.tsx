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
  Hourglass,
  Layers3,
  ListOrdered,
  Loader2,
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
  QueueSnapshot,
  SystemInfo,
  TaskRecord,
  TaskStatus,
  ToolInfo,
  ToolKnowledge,
  WebSearchResult,
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
  const [toolKnowledge, setToolKnowledge] = useState<ToolKnowledge[]>([]);
  const [applications, setApplications] = useState<ApplicationInfo[]>([]);
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [activeTask, setActiveTask] = useState<TaskRecord | null>(null);
  const [queue, setQueue] = useState<QueueSnapshot>({ running: null, pending: [] });
  const [request, setRequest] = useState("");
  const [loading, setLoading] = useState(false);
  // Which task is currently being approved/denied, to block duplicate clicks.
  const [resolvingId, setResolvingId] = useState<string | null>(null);
  const [phase, setPhase] = useState<string | null>(null);
  const [appsLoading, setAppsLoading] = useState(false);
  const [toolsLoading, setToolsLoading] = useState(false);
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
    if (section !== "tools" || toolKnowledge.length > 0 || toolsLoading) return;
    setToolsLoading(true);
    api.toolKnowledge()
      .then((result) => setToolKnowledge(result.tools))
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setToolsLoading(false));
  }, [section, toolKnowledge.length, toolsLoading]);

  // Poll the active task and the queue while anything is in flight, so the UI
  // reflects real step-by-step progress and shows what is still waiting.
  useEffect(() => {
    const inFlight = activeTask && ["PENDING", "RUNNING"].includes(activeTask.status);
    if (!inFlight) return;
    const timer = window.setInterval(() => {
      api.task(activeTask.id)
        .then((nextTask) => {
          setActiveTask(nextTask);
          setTasks((current) => [nextTask, ...current.filter((task) => task.id !== nextTask.id)]);
          setPhase(describePhase(nextTask));
        })
        .catch((reason: Error) => setError(reason.message));
      api.queue().then(setQueue).catch(() => undefined);
    }, 900);
    return () => window.clearInterval(timer);
  }, [activeTask]);

  const completedCount = useMemo(
    () => tasks.filter((task) => task.status === "COMPLETED").length,
    [tasks],
  );

  // Derive a human-readable phase from the active task whenever it changes.
  useEffect(() => {
    setPhase(activeTask ? describePhase(activeTask) : null);
  }, [activeTask]);

  async function submitTask(event: React.FormEvent) {
    event.preventDefault();
    const cleanRequest = request.trim();
    if (!cleanRequest || loading) return;
    setLoading(true);
    setPhase("Submitting to the local runtime...");
    setError(null);
    try {
      const task = await api.createTask(cleanRequest);
      setActiveTask(task);
      setTasks((current) => [task, ...current.filter((item) => item.id !== task.id)]);
      setRequest("");
      // Refresh the queue immediately so a second submission appears as queued.
      api.queue().then(setQueue).catch(() => undefined);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not submit task");
    } finally {
      setLoading(false);
      setPhase(null);
    }
  }

  async function resolveTask(decision: "approve" | "deny") {
    if (!activeTask) return;
    // Guard against duplicate authorization: ignore clicks while one is in
    // flight, and never a second request for the same task.
    if (resolvingId === activeTask.id) return;
    setResolvingId(activeTask.id);
    setPhase(decision === "approve" ? "Applying your approval..." : "Cancelling the action...");
    setError(null);
    try {
      const task = decision === "approve"
        ? await api.approveTask(activeTask.id)
        : await api.denyTask(activeTask.id);
      setActiveTask(task);
      setTasks((current) => [task, ...current.filter((item) => item.id !== task.id)]);
      api.queue().then(setQueue).catch(() => undefined);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Authorization failed");
    } finally {
      setResolvingId(null);
      setPhase(null);
      refreshTasks();
    }
  }

  const approveTask = () => resolveTask("approve");
  const denyTask = () => resolveTask("deny");

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
          {section === "command" ? <CommandWorkspace activeTask={activeTask} queue={queue} tasks={tasks} request={request} setRequest={setRequest} loading={loading} phase={phase} resolvingId={resolvingId} onSubmit={submitTask} onApprove={approveTask} onDeny={denyTask} onSelect={setActiveTask} completedCount={completedCount} /> : null}
          {section === "tasks" ? <TasksView tasks={tasks} activeTask={activeTask} onSelect={setActiveTask} /> : null}
          {section === "applications" ? <ApplicationsView applications={applications} loading={appsLoading} /> : null}
          {section === "tools" ? <ToolsView tools={tools} knowledge={toolKnowledge} loading={toolsLoading} /> : null}
          {section === "system" ? <SystemView system={system} health={health} /> : null}
          {section === "files" || section === "knowledge" || section === "memory" || section === "settings" ? <UnavailableView section={section} /> : null}
        </div>
      </main>
    </div>
  );
}

function CommandWorkspace({
  activeTask,
  queue,
  tasks,
  request,
  setRequest,
  loading,
  phase,
  resolvingId,
  onSubmit,
  onApprove,
  onDeny,
  onSelect,
  completedCount,
}: {
  activeTask: TaskRecord | null;
  queue: QueueSnapshot;
  tasks: TaskRecord[];
  request: string;
  setRequest: (value: string) => void;
  loading: boolean;
  phase: string | null;
  resolvingId: string | null;
  onSubmit: (event: FormEvent) => void;
  onApprove: () => void;
  onDeny: () => void;
  onSelect: (task: TaskRecord) => void;
  completedCount: number;
}) {
  const queuedTasks = queue.pending
    .map((id) => tasks.find((task) => task.id === id))
    .filter((task): task is TaskRecord => Boolean(task));
  const busy = loading || resolvingId !== null;
  return (
    <div className="workspace-grid">
      <section className="command-column">
        <div className="eyebrow"><span className="eyebrow-line" /> Command interface</div>
        <h1>What should Atlas<br /><span>take care of?</span></h1>
        <p className="lead">Describe a task in plain language. Atlas will plan it, request approval when needed, and report only verified execution state.</p>
        <form className={`command-box ${busy ? "command-box-busy" : ""}`} onSubmit={onSubmit} aria-busy={busy}>
          <div className="command-box-top"><span className="input-prompt">›</span><textarea value={request} onChange={(event) => setRequest(event.target.value)} placeholder="Ask Atlas to inspect, search, or act..." rows={3} disabled={busy} /></div>
          <div className="command-box-bottom"><span className="hint"><kbd>Enter</kbd> submit · <kbd>Shift Enter</kbd> new line</span><button className="send-button" disabled={busy || !request.trim()} title="Submit task">{loading ? <RotateCw size={17} className="spin" /> : <Send size={17} />}</button></div>
          {loading ? <div className="command-progress"><Loader2 size={15} className="spin" /> {phase ?? "Sending your request..."}</div> : null}
        </form>
        <QueuePanel queue={queue} tasks={queuedTasks} activeTaskId={activeTask?.id ?? null} onSelect={onSelect} />
        <div className="suggestion-row">
          <button onClick={() => setRequest("Inspect my system and report the current disk space")}>Inspect system <ArrowUpRight size={13} /></button>
          <button onClick={() => setRequest("Find all PDF files in the project")}>Find project PDFs <ArrowUpRight size={13} /></button>
          <button onClick={() => setRequest("Search my knowledge base for the latest project status")}>Search knowledge <ArrowUpRight size={13} /></button>
        </div>
        {activeTask ? <TaskPanel task={activeTask} phase={phase} resolving={resolvingId === activeTask.id} onApprove={onApprove} onDeny={onDeny} /> : <EmptyTaskState completedCount={completedCount} />}
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

function describePhase(task: TaskRecord): string {
  if (task.status === "WAITING_FOR_CONFIRMATION") return "Waiting for your approval.";
  if (task.status === "PENDING") return "Queued — waiting for the current task to finish.";
  const steps = task.plan?.steps ?? [];
  const current = steps.find((step) => step.status === "RUNNING");
  if (current) return `Running: ${current.name}`;
  const done = steps.filter((step) => step.status === "COMPLETED").length;
  if (steps.length) return `Step ${Math.min(done + 1, steps.length)} of ${steps.length}...`;
  return "Atlas is working through the plan step by step...";
}

function QueuePanel({
  queue,
  tasks,
  activeTaskId,
  onSelect,
}: {
  queue: QueueSnapshot;
  tasks: TaskRecord[];
  activeTaskId: string | null;
  onSelect: (task: TaskRecord) => void;
}) {
  // The queue is only meaningful when something is running or waiting.
  if (!queue.running && queue.pending.length === 0) return null;
  return (
    <section className="queue-panel">
      <div className="queue-header"><ListOrdered size={15} /><span>Task queue</span><em>{queue.running ? 1 + queue.pending.length : queue.pending.length} in flight</em></div>
      <p className="queue-note">Atlas runs one task at a time so actions never overlap.</p>
      {tasks.map((task, index) => (
        <button key={task.id} className={`queue-item ${task.id === activeTaskId ? "queue-item-active" : ""}`} onClick={() => onSelect(task)}>
          <span className="queue-position">{task.id === queue.running ? <Zap size={13} /> : index + 1}</span>
          <span className="queue-request">{task.request}</span>
          <small>{task.id === queue.running ? "Running" : "Queued"}</small>
        </button>
      ))}
      {queue.pending.length > 0 ? <div className="queue-waiting"><Hourglass size={13} /> {queue.pending.length} waiting</div> : null}
    </section>
  );
}

function TaskPanel({ task, phase, resolving, onApprove, onDeny }: { task: TaskRecord; phase: string | null; resolving: boolean; onApprove: () => void; onDeny: () => void }) {
  const waiting = task.status === "WAITING_FOR_CONFIRMATION";
  const running = task.status === "RUNNING" || task.status === "PENDING";
  return (
    <section className={`task-panel ${running ? "task-panel-live" : ""}`}>
      <div className="section-heading"><div><span className="eyebrow">Current task</span><h2>{statusLabels[task.status]}</h2></div><StatusPill status={task.status} /></div>
      <p className="task-request">{task.request}</p>
      {running ? <div className="task-progress"><Loader2 size={15} className="spin" /> <span>{phase ?? "Atlas is working through the plan step by step..."}</span></div> : null}
      {waiting ? <div className="approval-box"><div><LockKeyhole size={18} /><div><strong>Atlas is ready to act</strong><span>This action changes your computer. Review the plan and approve it to continue.</span></div></div><div className="approval-actions"><button className="button-muted" onClick={onDeny} disabled={resolving}><X size={15} /> Deny</button><button className="button-primary" onClick={onApprove} disabled={resolving}>{resolving ? <><Loader2 size={15} className="spin" /> Working...</> : <><Check size={15} /> Approve action</>}</button></div></div> : null}
      {task.plan ? <div className="plan-list"><div className="plan-label">Execution plan</div>{task.plan.steps.map((step) => <div className="plan-step-group" key={step.id}><div className="plan-step"><span className={`step-icon ${step.status.toLowerCase()}`}><StepIcon status={step.status} /></span><span>{step.name}</span><small>{step.status.replaceAll("_", " ").toLowerCase()}</small></div>{typeof step.metadata.capability === "string" ? <div className="plan-detail"><strong>{step.metadata.capability}</strong>{Array.isArray(step.metadata.candidates) ? <span> Candidates: {step.metadata.candidates.join(", ")}</span> : null}{typeof step.metadata.command === "string" ? <code>{step.metadata.command}</code> : null}</div> : null}</div>)}</div> : null}
      <ReasoningPanel task={task} />
      {task.response ? <div className="result-box"><span>Atlas result</span><p>{task.response}</p></div> : null}
      {task.tool_calls.length ? <div className="tool-results"><div className="plan-label">Tool output</div>{task.tool_calls.map((call, index) => <details key={`${call.tool ?? "tool"}-${index}`}><summary>{call.tool ?? "Tool"} <span>{call.status ?? "unknown"}</span></summary><pre>{formatToolOutput(call.output, call.status)}</pre></details>)}</div> : null}
      <WebResults calls={task.tool_calls} webResults={task.web_results} />
      {task.web_sources.length ? <div className="source-list"><div className="plan-label">Sources</div>{task.web_sources.map((source) => <a key={source} href={source} target="_blank" rel="noreferrer">{source}</a>)}</div> : null}
      {task.errors.length ? <div className="failure-box"><CircleAlert size={15} /> {task.errors.join(" ")}</div> : null}
    </section>
  );
}

function ReasoningPanel({ task }: { task: TaskRecord }) {
  const steps = task.reasoning ?? [];
  const provenance = task.provenance ?? [];
  const citations = task.citations ?? [];
  if (!steps.length && !task.response_mode && !provenance.length && !citations.length && !task.evidence) return null;
  return (
    <div className="plan-list" aria-label="Reasoning activity">
      <div className="plan-label">Reasoning activity</div>
      {steps.map((step, index) => (
        <div className="plan-step-group" key={`${step.stage}-${index}`}>
          <div className="plan-step"><span className="step-icon"><BrainCircuit size={13} /></span><span>{step.stage.replaceAll("_", " ").toLowerCase()}</span><small>{step.iteration > 0 ? `Pass ${step.iteration}` : ""}</small></div>
          {step.detail ? <div className="plan-detail">{step.detail}</div> : null}
        </div>
      ))}
      {task.response_mode ? <div className="plan-detail"><strong>Response mode</strong> {task.response_mode.replaceAll("_", " ")}</div> : null}
      {provenance.length ? <div className="plan-detail"><strong>Answer sources</strong> {provenance.join(", ")}</div> : null}
      {task.evidence ? <div className="plan-detail"><strong>Evidence</strong> {task.evidence.count} item{task.evidence.count === 1 ? "" : "s"}{task.evidence.sources.length ? ` from ${task.evidence.sources.join(", ")}` : ""}</div> : null}
      {citations.length ? <div className="source-list"><div className="plan-label">Citations</div>{citations.map((citation) => <div key={citation}>{/^https?:\/\//i.test(citation) ? <a href={citation} target="_blank" rel="noreferrer">{citation}</a> : <span>{citation}</span>}</div>)}</div> : null}
    </div>
  );
}

function WebResults({ calls, webResults }: { calls: Array<{ tool?: string; output?: unknown }>; webResults?: Array<{ title: string; url: string; snippet: string; source: string; thumbnail_url?: string }> }) {
  const toolResults = calls
    .filter((call) => call.tool === "web.search")
    .flatMap((call) => {
      const output = call.output as { results?: WebSearchResult[] } | undefined;
      return Array.isArray(output?.results) ? output.results : [];
    });
  const allResults = [...toolResults, ...(webResults || [])];
  if (!allResults.length) return null;
  return <div className="web-results"><div className="plan-label">Web results</div><div className="web-result-grid">{allResults.map((result) => <a className="web-result-card" href={result.url} target="_blank" rel="noreferrer" key={result.url}><div className="web-thumb">{result.thumbnail_url ? <img src={result.thumbnail_url} alt="" loading="lazy" onError={(event) => { event.currentTarget.style.display = "none"; }} /> : <span>{result.source.slice(0, 1)}</span>}<span className="web-source">{result.source}</span></div><div className="web-result-copy"><strong>{result.title}</strong><p>{result.snippet || "Open source"}</p><small>{result.url}</small></div></a>)}</div></div>;
}

function formatToolOutput(output: unknown, status?: string) {
  if (typeof output === "string") return output;
  if (output === undefined || output === null) {
    if (status === "confirmation_required") return "Waiting for your approval.";
    if (status === "PENDING") return "Pending.";
    return "No output returned.";
  }
  return JSON.stringify(output, null, 2);
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

function ToolsView({ tools, knowledge, loading }: { tools: ToolInfo[]; knowledge: ToolKnowledge[]; loading: boolean }) {
  const [query, setQuery] = useState("");
  const [candidates, setCandidates] = useState<Array<{ tool: string; reason: string; risk_level: string; read_only: boolean }>>([]);
  const [discovering, setDiscovering] = useState(false);

  async function discover(event: FormEvent) {
    event.preventDefault();
    if (!query.trim() || discovering) return;
    setDiscovering(true);
    try {
      const result = await api.discoverTools(query.trim());
      setCandidates(result.candidates);
    } catch (reason) {
      setCandidates([]);
    } finally {
      setDiscovering(false);
    }
  }

  return <PageFrame eyebrow="Runtime / capability registry" title="Available tools" description="Inspect live runtime capabilities, search the authoritative PowerShell knowledge catalog, and preview tool candidates without executing them.">
    <form className="tool-discovery" onSubmit={discover}>
      <Search size={17} />
      <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find tools for inspecting memory, services, or files" />
      <button type="submit" disabled={discovering || !query.trim()} title="Discover tools">{discovering ? <RotateCw size={15} className="spin" /> : <Search size={15} />}</button>
    </form>
    {candidates.length ? <div className="candidate-panel"><div className="plan-label">Discovery candidates</div>{candidates.map((candidate) => <div className="candidate-row" key={candidate.tool}><div><strong>{candidate.tool}</strong><span>{candidate.reason}</span></div><small>{candidate.read_only ? "read only" : candidate.risk_level}</small></div>)}</div> : null}
    <div className="tool-section-label">Registered runtime tools</div>
    <div className="tool-grid">{tools.map((tool) => <div className="tool-card" key={tool.name}><div className="tool-card-top"><span className="tool-symbol"><Zap size={16} /></span><span className={`risk risk-${tool.risk_level}`}>{tool.permission_level.replaceAll("_", " ")}</span></div><strong>{tool.name}</strong><p>{tool.description}</p><small>{tool.category}</small></div>)}{!tools.length ? <UnavailableMessage title="Tool registry unavailable" detail="Start the Atlas API to inspect real capabilities." /> : null}</div>
    <div className="tool-section-label">PowerShell knowledge catalog</div>
    <div className="knowledge-table">{loading ? <LoadingState /> : knowledge.map((record) => <details key={record.name}><summary><span><strong>{record.name}</strong><small>{record.category}</small></span><em>{record.read_only ? "SAFE / READ ONLY" : record.risk_level.toUpperCase()}</em></summary><div className="knowledge-detail"><p>{record.description}</p><span>{record.purpose.join(" · ")}</span>{record.examples.length ? <pre>{record.examples.join("\n")}</pre> : null}</div></details>)}</div>
  </PageFrame>;
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
