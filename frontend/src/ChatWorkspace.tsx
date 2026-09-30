import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronRight,
  CircleAlert,
  Copy,
  Loader2,
  Paperclip,
  Pencil,
  Plus,
  RotateCcw,
  Search,
  Send,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { api, streamMessage } from "./services/api";
import type {
  Conversation,
  ConversationAttachment,
  ConversationMessage,
  ConversationSearchHit,
  ConversationStreamEvent,
  MemoryRecord,
  ResponseKind,
} from "./types";

/**
 * The conversational Atlas workspace.
 *
 * It is a client of the Conversation Runtime, not a second agent: every message
 * is sent to `/api/messages/stream`, and everything shown in the transcript
 * comes from the turn that was actually persisted. Tool activity is displayed
 * from real recorded tool calls, and a turn's state (completed / cancelled /
 * failed / awaiting approval) is the state the backend recorded.
 */

const kindLabels: Record<ResponseKind, string> = {
  conversation: "Conversation",
  retrieval: "Retrieved",
  research: "Research",
  task: "Task",
  computer: "Computer action",
  hybrid: "Research + action",
  clarification: "Needs clarification",
  memory: "From memory",
  system: "System",
  error: "Error",
};

interface PendingTurn {
  conversationId: string | null;
  text: string;
  tokens: string;
  activity: string[];
  turnId: string | null;
  cancelled: boolean;
}

export function ChatWorkspace({ modelLabel }: { modelLabel: string }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<PendingTurn | null>(null);
  const [search, setSearch] = useState("");
  const [hits, setHits] = useState<ConversationSearchHit[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [attachments, setAttachments] = useState<ConversationAttachment[]>([]);
  const [attachmentPath, setAttachmentPath] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);

  const refreshConversations = useCallback(async () => {
    try {
      const result = await api.conversations();
      setConversations(result.conversations);
      return result.conversations;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load conversations");
      return [];
    }
  }, []);

  const openConversation = useCallback(async (id: string) => {
    setActiveId(id);
    setError(null);
    try {
      const result = await api.conversationMessages(id);
      setMessages(result.messages);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load the conversation");
    }
  }, []);

  useEffect(() => {
    void (async () => {
      const list = await refreshConversations();
      if (list.length) await openConversation(list[0].id);
    })();
  }, [refreshConversations, openConversation]);

  // Keep the newest turn in view as tokens arrive.
  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight });
  }, [messages, pending?.tokens, pending?.activity.length]);

  useEffect(() => {
    if (!search.trim()) {
      setHits([]);
      return;
    }
    const timer = window.setTimeout(() => {
      api
        .searchConversations(search.trim())
        .then((result) => setHits(result.results))
        .catch(() => setHits([]));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [search]);

  async function newConversation() {
    setError(null);
    try {
      const conversation = await api.createConversation();
      setConversations((current) => [conversation, ...current]);
      setActiveId(conversation.id);
      setMessages([]);
      setDraft("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not start a conversation");
    }
  }

  async function deleteConversation(id: string) {
    setError(null);
    try {
      await api.deleteConversation(id);
      const remaining = conversations.filter((item) => item.id !== id);
      setConversations(remaining);
      if (activeId === id) {
        setActiveId(null);
        setMessages([]);
        if (remaining.length) await openConversation(remaining[0].id);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not delete the conversation");
    }
  }

  async function commitRename(id: string) {
    const title = renameDraft.trim();
    setRenamingId(null);
    if (!title) return;
    try {
      const updated = await api.renameConversation(id, title);
      setConversations((current) => current.map((item) => (item.id === id ? updated : item)));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not rename the conversation");
    }
  }

  function addAttachment() {
    const path = attachmentPath.trim();
    if (!path) return;
    // Only a path is sent. The server reads the file through its own
    // permission-gated file tools, exactly as it would for any other read.
    const name = path.split(/[\\/]/).filter(Boolean).pop() ?? path;
    setAttachments((current) =>
      current.some((item) => item.path === path) ? current : [...current, { name, path }],
    );
    setAttachmentPath("");
  }

  function removeAttachment(path: string) {
    setAttachments((current) => current.filter((item) => item.path !== path));
  }

  async function send(explicit?: { text: string; attachments?: ConversationAttachment[] }) {
    const text = (explicit?.text ?? draft).trim();
    const files = explicit?.attachments ?? attachments;
    if (!text || pending) return;
    setError(null);
    setDraft("");
    setAttachments([]);
    const turn: PendingTurn = {
      conversationId: activeId,
      text,
      tokens: "",
      activity: [],
      turnId: null,
      cancelled: false,
    };
    setPending(turn);

    const controller = new AbortController();
    abortRef.current = controller;

    const onEvent = (event: ConversationStreamEvent) => {
      if (event.type === "open") {
        const turnId = typeof event.data.turn_id === "string" ? event.data.turn_id : null;
        setPending((current) => (current ? { ...current, turnId } : current));
        return;
      }
      if (event.type === "token") {
        const piece = typeof event.data.text === "string" ? event.data.text : "";
        setPending((current) => (current ? { ...current, tokens: current.tokens + piece } : current));
        return;
      }
      if (event.type === "tool_started") {
        // A real tool start names the capability. The routing notice (no tool)
        // explains what Atlas decided to do, so it is shown as that.
        const line =
          typeof event.data.tool === "string"
            ? `${event.data.tool} â€” started`
            : String(event.data.detail ?? "working");
        setPending((current) =>
          current ? { ...current, activity: [...current.activity, line] } : current,
        );
        return;
      }
      if (event.type === "tool_completed") {
        const tool = typeof event.data.tool === "string" ? event.data.tool : "tool";
        const status = String(event.data.status ?? (event.data.success ? "completed" : "failed"));
        setPending((current) =>
          current ? { ...current, activity: [...current.activity, `${tool} â€” ${status}`] } : current,
        );
        return;
      }
      if (event.type === "observation") {
        const summary = String(event.data.summary ?? "observed the screen");
        setPending((current) =>
          current ? { ...current, activity: [...current.activity, summary] } : current,
        );
        return;
      }
      if (event.type === "verification") {
        const tool = typeof event.data.tool === "string" ? event.data.tool : "action";
        const status = String(event.data.status ?? "checked");
        setPending((current) =>
          current ? { ...current, activity: [...current.activity, `Verified ${tool} â€” ${status}`] } : current,
        );
        return;
      }
      if (event.type === "attachment") {
        const names = Array.isArray(event.data.names) ? event.data.names.map(String) : [];
        if (names.length) {
          setPending((current) =>
            current ? { ...current, activity: [...current.activity, `Read ${names.join(", ")}`] } : current,
          );
        }
      }
    };

    try {
      await streamMessage(text, {
        conversationId: activeId,
        attachments: files,
        onEvent,
        signal: controller.signal,
      });
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === "AbortError")) {
        setError(reason instanceof Error ? reason.message : "The turn failed");
      }
    } finally {
      abortRef.current = null;
      // Re-read the transcript from the server: the persisted turn is the
      // authority, so what is displayed is what was actually stored. A streamed
      // answer is never re-typed into state by hand.
      try {
        let conversationsNow = conversations;
        if (!activeId) {
          conversationsNow = await refreshConversations();
        }
        const target = activeId ?? conversationsNow[0]?.id ?? null;
        if (target) {
          setActiveId(target);
          const result = await api.conversationMessages(target);
          setMessages(result.messages);
          setConversations((current) =>
            current.map((item) =>
              item.id === target
                ? {
                    ...item,
                    message_count: result.messages.length,
                    title: conversationsNow.find((c) => c.id === target)?.title ?? item.title,
                  }
                : item,
            ),
          );
        }
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Could not refresh the conversation");
      }
      setPending(null);
    }
  }

  async function stop() {
    const turnId = pending?.turnId;
    if (!turnId) {
      abortRef.current?.abort();
      return;
    }
    setPending((current) => (current ? { ...current, cancelled: true } : current));
    try {
      await api.cancelTurn(turnId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel the turn");
    }
  }

  /**
   * Edit a past message: the original wording is put back in the composer so it
   * can be changed and sent as a new turn. Nothing is silently rewritten in the
   * stored transcript â€” what Atlas actually said stays visible.
   */
  function editMessage(message: ConversationMessage) {
    setDraft(message.content);
  }

  /**
   * Regenerate: ask the same request again. This is a real new turn through the
   * real runtime, and it is appended to the conversation rather than replacing
   * history, so the earlier answer remains auditable.
   */
  function regenerate(message: ConversationMessage) {
    const index = messages.findIndex((item) => item.id === message.id);
    if (index <= 0) return;
    const previous = messages.slice(0, index).reverse().find((item) => item.role === "user");
    if (!previous) return;
    void send({ text: previous.content, attachments: previous.attachments ?? [] });
  }

  const lastAssistantIndex = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index].role === "assistant") return index;
    }
    return -1;
  }, [messages]);

  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeId) ?? null,
    [conversations, activeId],
  );

  return (
    <div className="chat-shell">
      <aside className="chat-sidebar">
        <button className="new-task" onClick={newConversation}>
          <Plus size={16} /> New chat
        </button>
        <label className="chat-search">
          <Search size={14} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search conversations"
          />
          {search ? (
            <button type="button" onClick={() => setSearch("")} title="Clear search">
              <X size={13} />
            </button>
          ) : null}
        </label>

        {hits.length ? (
          <div className="chat-hits">
            <div className="chat-list-label">Matches</div>
            {hits.slice(0, 8).map((hit) => (
              <button
                key={`${hit.conversation_id}-${hit.message_id || hit.match}`}
                className="chat-hit"
                onClick={() => openConversation(hit.conversation_id)}
              >
                <strong>{hit.title || "Untitled"}</strong>
                <span>{hit.excerpt}</span>
                <small>{hit.match}</small>
              </button>
            ))}
          </div>
        ) : null}

        <div className="chat-list-label">Conversations</div>
        <div className="chat-list">
          {conversations.length === 0 ? (
            <p className="chat-empty-note">No conversations yet.</p>
          ) : (
            conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`chat-item ${conversation.id === activeId ? "chat-item-active" : ""}`}
              >
                {renamingId === conversation.id ? (
                  <input
                    className="chat-rename"
                    value={renameDraft}
                    autoFocus
                    onChange={(event) => setRenameDraft(event.target.value)}
                    onBlur={() => commitRename(conversation.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") commitRename(conversation.id);
                      if (event.key === "Escape") setRenamingId(null);
                    }}
                  />
                ) : (
                  <button className="chat-item-main" onClick={() => openConversation(conversation.id)}>
                    <strong>{conversation.title || "Untitled"}</strong>
                    <small>{conversation.message_count} messages</small>
                  </button>
                )}
                <div className="chat-item-actions">
                  <button
                    title="Rename"
                    onClick={() => {
                      setRenamingId(conversation.id);
                      setRenameDraft(conversation.title || "");
                    }}
                  >
                    <Pencil size={12} />
                  </button>
                  <button title="Delete" onClick={() => deleteConversation(conversation.id)}>
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
        <div className="chat-model-chip">{modelLabel}</div>
      </aside>

      <section className="chat-main">
        <header className="chat-header">
          <div>
            <span className="eyebrow">Conversation</span>
            <h2>{activeConversation?.title || "New conversation"}</h2>
          </div>
          {activeConversation?.summary ? (
            <details className="chat-summary">
              <summary>Summary</summary>
              <pre>{activeConversation.summary}</pre>
            </details>
          ) : null}
        </header>

        {error ? <div className="error-strip"><CircleAlert size={15} /> {error}</div> : null}

        <div className="chat-transcript" ref={transcriptRef}>
          {messages.length === 0 && !pending ? (
            <div className="chat-welcome">
              <h3>Talk to Atlas.</h3>
              <p>
                Ask a question, request research, or ask Atlas to do something on this machine. I keep the
                conversation, remember what matters, and show you what I actually did.
              </p>
              <div className="chat-suggestions">
                {[
                  "What is Docker?",
                  "Research the latest Python changes.",
                  "Open Chrome.",
                  "Remember that I prefer local-first architecture.",
                ].map((suggestion) => (
                  <button key={suggestion} onClick={() => setDraft(suggestion)}>
                    {suggestion} <ChevronRight size={13} />
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              onEdit={editMessage}
              onRegenerate={regenerate}
              canRegenerate={index === lastAssistantIndex && !pending}
            />
          ))}

          {pending ? (
            <div className="chat-message chat-message-assistant">
              <div className="chat-role">Atlas</div>
              <div className="chat-bubble">
                {pending.tokens ? (
                  <p className="chat-text">{pending.tokens}</p>
                ) : (
                  <p className="chat-text chat-thinking">
                    <Loader2 size={14} className="spin" /> Workingâ€¦
                  </p>
                )}
              </div>
              {pending.activity.length ? (
                <details className="chat-activity" open>
                  <summary>Activity</summary>
                  <ul>
                    {pending.activity.map((line, index) => (
                      <li key={`${line}-${index}`}>{line}</li>
                    ))}
                  </ul>
                </details>
              ) : null}
            </div>
          ) : null}
        </div>

        <form
          className={`chat-composer ${pending ? "chat-composer-busy" : ""}`}
          onSubmit={(event) => {
            event.preventDefault();
            void send();
          }}
        >
          <textarea
            value={draft}
            rows={2}
            disabled={Boolean(pending)}
            placeholder="Message Atlasâ€¦"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
          />
          {attachments.length ? (
            <div className="chat-attachment-list">
              {attachments.map((attachment) => (
                <span key={attachment.path ?? attachment.name} className="chat-attachment">
                  <Paperclip size={12} /> {attachment.name}
                  <button type="button" onClick={() => removeAttachment(attachment.path ?? "")} title="Remove">
                    <X size={11} />
                  </button>
                </span>
              ))}
            </div>
          ) : null}
          {!pending ? (
            <div className="chat-attach-row">
              <Paperclip size={13} />
              <input
                value={attachmentPath}
                placeholder="Attach a file by path (Atlas reads it with the same permissions as any file)"
                onChange={(event) => setAttachmentPath(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    addAttachment();
                  }
                }}
              />
              <button type="button" onClick={addAttachment} disabled={!attachmentPath.trim()}>
                Attach
              </button>
            </div>
          ) : null}
          <div className="chat-composer-actions">
            <span className="hint"><kbd>Enter</kbd> send Â· <kbd>Shift Enter</kbd> newline</span>
            {pending ? (
              <button type="button" className="button-muted" onClick={() => void stop()}>
                <Square size={14} /> Stop
              </button>
            ) : (
              <button type="submit" className="send-button" disabled={!draft.trim()} title="Send">
                <Send size={16} />
              </button>
            )}
          </div>
        </form>
      </section>
    </div>
  );
}

function MessageBubble({
  message,
  onEdit,
  onRegenerate,
  canRegenerate,
}: {
  message: ConversationMessage;
  onEdit: (message: ConversationMessage) => void;
  onRegenerate: (message: ConversationMessage) => void;
  canRegenerate: boolean;
}) {
  const kind = (message.metadata?.response_kind || "") as ResponseKind | "";
  const activity = Array.isArray(message.metadata?.activity) ? message.metadata.activity : [];
  const cites = message.citations ?? [];
  const state = message.execution_state;
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className={`chat-message chat-message-${message.role}`}>
      <div className="chat-role">
        {message.role === "user" ? "You" : "Atlas"}
        {kind ? <em>{kindLabels[kind] ?? kind}</em> : null}
        {state !== "completed" ? <em className={`chat-state chat-state-${state}`}>{state.replaceAll("_", " ")}</em> : null}
      </div>
      <div className="chat-bubble">
        <p className="chat-text">{message.content}</p>
      </div>

      {message.attachments?.length ? (
        <div className="chat-attachments">
          {message.attachments.map((attachment, index) => (
            <span key={`${attachment.name}-${index}`} className="chat-attachment">
              {attachment.name}
              {attachment.error ? <em> â€” {attachment.error}</em> : null}
            </span>
          ))}
        </div>
      ) : null}

      <div className="chat-message-actions">
        <button type="button" onClick={copy} title="Copy">
          {copied ? <Check size={12} /> : <Copy size={12} />} {copied ? "Copied" : "Copy"}
        </button>
        {message.role === "user" ? (
          <button type="button" onClick={() => onEdit(message)} title="Edit and resend">
            <Pencil size={12} /> Edit
          </button>
        ) : null}
        {message.role === "assistant" && canRegenerate ? (
          <button type="button" onClick={() => onRegenerate(message)} title="Ask again">
            <RotateCcw size={12} /> Regenerate
          </button>
        ) : null}
      </div>

      {activity.length ? (
        <details className="chat-activity">
          <summary>
            Activity <span>{activity.length}</span>
          </summary>
          <ul>
            {activity.map((line, index) => (
              <li key={`${line}-${index}`}>{line}</li>
            ))}
          </ul>
        </details>
      ) : null}

      {message.tool_calls?.length ? (
        <details className="chat-activity">
          <summary>
            Tool calls <span>{message.tool_calls.length}</span>
          </summary>
          <ul>
            {message.tool_calls.map((call, index) => (
              <li key={`${call.tool ?? "tool"}-${index}`}>
                {call.tool ?? "tool"} â€” {call.status ?? "unknown"}
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {cites.length ? (
        <div className="chat-citations">
          {cites.map((citation) => (
            <a key={citation} href={citation} target="_blank" rel="noreferrer">
              {citation}
            </a>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** The memory view: view, search, add, edit, delete, clear, disable. */
export function MemoryView() {
  const [records, setRecords] = useState<MemoryRecord[]>([]);
  const [status, setStatus] = useState<{ enabled: boolean; count: number } | null>(null);
  const [query, setQuery] = useState("");
  const [draft, setDraft] = useState("");
  const [kind, setKind] = useState("preference");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (searchQuery = "") => {
    try {
      const result = await api.memories(searchQuery);
      setRecords(result.memories);
      setStatus({ enabled: result.enabled, count: result.count });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load memory");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section className="page-frame">
      <div className="page-head">
        <span className="eyebrow">Memory / long-term</span>
        <h1>What Atlas remembers</h1>
        <p className="lead">
          Only durable information is stored: stable preferences, explicit instructions, important
          projects, and remembered facts. Ordinary conversation is never turned into permanent memory,
          and everything here can be edited, deleted, or disabled.
        </p>
      </div>

      {error ? <div className="error-strip"><CircleAlert size={15} /> {error}</div> : null}

      <div className="memory-toolbar">
        <label className="chat-search">
          <Search size={14} />
          <input
            value={query}
            placeholder="Search memory"
            onChange={(event) => {
              setQuery(event.target.value);
              void load(event.target.value);
            }}
          />
        </label>
        <button
          className="button-muted"
          onClick={async () => {
            if (!status) return;
            const next = await api.setMemoryEnabled(!status.enabled);
            setStatus({ enabled: next.enabled, count: next.count });
          }}
        >
          {status?.enabled ? "Disable memory" : "Enable memory"}
        </button>
        <button
          className="button-muted"
          onClick={async () => {
            await api.clearMemory();
            await load(query);
          }}
        >
          Clear all
        </button>
      </div>

      <form
        className="memory-add"
        onSubmit={async (event) => {
          event.preventDefault();
          if (!draft.trim()) return;
          try {
            await api.createMemory(draft.trim(), kind);
            setDraft("");
            await load(query);
          } catch (reason) {
            setError(reason instanceof Error ? reason.message : "Could not store that memory");
          }
        }}
      >
        <input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Something Atlas should remember" />
        <select value={kind} onChange={(event) => setKind(event.target.value)}>
          <option value="preference">Preference</option>
          <option value="instruction">Instruction</option>
          <option value="fact">Fact</option>
          <option value="project">Project</option>
          <option value="workflow">Workflow</option>
        </select>
        <button className="button-primary" type="submit" disabled={!draft.trim() || status?.enabled === false}>
          Add
        </button>
      </form>

      <div className="memory-list">
        {records.length === 0 ? (
          <p className="chat-empty-note">Nothing remembered yet.</p>
        ) : (
          records.map((record) => (
            <div className="memory-row" key={record.id}>
              <div className="memory-row-main">
                <span className={`memory-kind memory-kind-${record.kind}`}>{record.kind}</span>
                {editingId === record.id ? (
                  <input
                    value={editDraft}
                    autoFocus
                    onChange={(event) => setEditDraft(event.target.value)}
                    onKeyDown={async (event) => {
                      if (event.key === "Enter") {
                        await api.updateMemory(record.id, editDraft);
                        setEditingId(null);
                        await load(query);
                      }
                      if (event.key === "Escape") setEditingId(null);
                    }}
                  />
                ) : (
                  <span>{record.text}</span>
                )}
              </div>
              <div className="memory-row-actions">
                {editingId === record.id ? (
                  <button
                    onClick={async () => {
                      await api.updateMemory(record.id, editDraft);
                      setEditingId(null);
                      await load(query);
                    }}
                  >
                    Save
                  </button>
                ) : (
                  <button
                    onClick={() => {
                      setEditingId(record.id);
                      setEditDraft(record.text);
                    }}
                    title="Edit"
                  >
                    <Pencil size={12} />
                  </button>
                )}
                <button
                  title="Forget"
                  onClick={async () => {
                    await api.deleteMemory(record.id);
                    await load(query);
                  }}
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

