import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronRight,
  CircleAlert,
  Copy,
  Loader2,
  LockKeyhole,
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
  // The last state the backend actually reported for this turn. "running" is
  // only shown while a turn is genuinely streaming; the terminal event
  // (assistant_completed / cancelled / error) replaces it truthfully.
  state: "running" | "waiting_for_confirmation" | "cancelled" | "failed" | "completed";
  // True only between the assistant_started event and its first token/activity:
  // before that, Atlas is routing the request, not composing an answer.
  started: boolean;
}

export function ChatWorkspace({ modelLabel }: { modelLabel: string }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<PendingTurn | null>(null);
  const [search, setSearch] = useState("");
  const [hits, setHits] = useState<ConversationSearchHit[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [attachments, setAttachments] = useState<ConversationAttachment[]>([]);
  const [attachmentPath, setAttachmentPath] = useState("");
  // The task id of a confirmation turn currently being approved/denied, so the
  // controls can be disabled and cannot be double-submitted.
  const [resolvingConfirmation, setResolvingConfirmation] = useState<string | null>(null);
  // True while a conversation's transcript is being fetched, so the welcome
  // panel is not shown for a conversation that simply has not loaded yet.
  const [loadingConversation, setLoadingConversation] = useState(false);
  // True from the moment a stop is requested until the backend has actually
  // reported the turn's terminal state. It reflects the real request, not a
  // fabricated progress state.
  const [stopping, setStopping] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  // Monotonic id for the latest conversation open. A slower response for an
  // earlier click must never overwrite the transcript of the conversation the
  // user actually switched to.
  const openRequestRef = useRef(0);
  // The conversation the user is currently viewing. Used to avoid clobbering a
  // switched-to conversation when an in-flight turn finishes.
  const activeIdRef = useRef<string | null>(null);

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
    const requestId = (openRequestRef.current += 1);
    setActiveId(id);
    activeIdRef.current = id;
    setError(null);
    setMessages([]);
    setLoadingConversation(true);
    try {
      const result = await api.conversationMessages(id);
      // Ignore a response that is no longer the latest request: the user has
      // switched (or switched back) since it was issued.
      if (requestId !== openRequestRef.current) return;
      setMessages(result.messages);
    } catch (reason) {
      if (requestId !== openRequestRef.current) return;
      setError(reason instanceof Error ? reason.message : "Could not load the conversation");
    } finally {
      if (requestId === openRequestRef.current) setLoadingConversation(false);
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
      setSearching(false);
      return;
    }
    let ignore = false;
    // A search is in progress from the moment the query changes, so an empty
    // result set is not mistaken for "no matches" while the request is pending.
    setSearching(true);
    const timer = window.setTimeout(() => {
      api
        .searchConversations(search.trim())
        .then((result) => {
          // A slower response for an earlier query must not replace the hits of
          // the query the user is on now.
          if (!ignore) setHits(result.results);
        })
        .catch(() => {
          if (!ignore) setHits([]);
        })
        .finally(() => {
          if (!ignore) setSearching(false);
        });
    }, 250);
    return () => {
      ignore = true;
      window.clearTimeout(timer);
    };
  }, [search]);

  async function newConversation() {
    setError(null);
    try {
      const conversation = await api.createConversation();
      setConversations((current) => [conversation, ...current]);
      // A newly created conversation is the latest open; any in-flight open for
      // an earlier conversation must not overwrite it.
      openRequestRef.current += 1;
      setActiveId(conversation.id);
      activeIdRef.current = conversation.id;
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
      if (activeIdRef.current === id) {
        openRequestRef.current += 1;
        setActiveId(null);
        activeIdRef.current = null;
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
    // Bind the turn to the conversation that is open right now. If the user
    // navigates elsewhere while it streams, the turn still lands in the right
    // conversation and the view is not dragged back.
    const turnConversationId = activeId;
    setError(null);
    setDraft("");
    setAttachments([]);
    const turn: PendingTurn = {
      conversationId: turnConversationId,
      text,
      tokens: "",
      activity: [],
      turnId: null,
      cancelled: false,
      state: "running",
      started: false,
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
      if (event.type === "assistant_started") {
        setPending((current) => (current ? { ...current, started: true } : current));
        return;
      }
      if (event.type === "token") {
        const piece = typeof event.data.text === "string" ? event.data.text : "";
        setPending((current) =>
          current ? { ...current, started: true, tokens: current.tokens + piece } : current,
        );
        return;
      }
      if (event.type === "assistant_completed") {
        // The backend reports the recorded terminal state. A turn that paused for
        // approval is NOT "working": it is waiting on the user, and is labelled
        // that way instead of showing a spinner that will never end on its own.
        const state = String(event.data.execution_state ?? "completed");
        setPending((current) =>
          current ? { ...current, state: state === "waiting_for_confirmation" ? "waiting_for_confirmation" : "completed" } : current,
        );
        return;
      }
      if (event.type === "cancelled") {
        setPending((current) => (current ? { ...current, state: "cancelled", cancelled: true } : current));
        return;
      }
      if (event.type === "tool_started") {
        // A real tool start names the capability. The routing notice (no tool)
        // explains what Atlas decided to do, so it is shown as that.
        const line =
          typeof event.data.tool === "string"
            ? `${event.data.tool} — started`
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
          current ? { ...current, activity: [...current.activity, `${tool} — ${status}`] } : current,
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
          current ? { ...current, activity: [...current.activity, `Verified ${tool} — ${status}`] } : current,
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
        return;
      }
      if (event.type === "error") {
        // A real streamed error must be surfaced, not swallowed: the terminal
        // status still comes from the refetched transcript, but the reason the
        // turn failed would otherwise be invisible.
        const detail =
          typeof event.data.detail === "string"
            ? event.data.detail
            : typeof event.data.text === "string"
              ? event.data.text
              : "The turn failed";
        setError(detail);
        setPending((current) => (current ? { ...current, state: "failed" } : current));
      }
    };

    try {
      await streamMessage(text, {
        conversationId: turnConversationId,
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
        if (!turnConversationId) {
          conversationsNow = await refreshConversations();
        }
        const target = turnConversationId ?? conversationsNow[0]?.id ?? null;
        if (target) {
          const result = await api.conversationMessages(target);
          // Adopt the transcript and, for a first message in a brand-new chat,
          // the new conversation id — but only if the user is still where they
          // were when they sent. If they switched away while it streamed, their
          // current view must not be replaced or navigated away from.
          if (activeIdRef.current === turnConversationId) {
            if (!turnConversationId) {
              setActiveId(target);
              activeIdRef.current = target;
            }
            setMessages(result.messages);
          }
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
      setStopping(false);
    }
  }

  async function stop() {
    const turnId = pending?.turnId;
    if (!turnId) {
      // The turn has not opened yet, so there is no backend turn to cancel.
      // Aborting the request stops the client from waiting on it; the stream
      // itself is what the user is asking to stop.
      abortRef.current?.abort();
      return;
    }
    setStopping(true);
    try {
      await api.cancelTurn(turnId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel the turn");
    } finally {
      // The cancelled state shown afterwards is the state the backend records
      // on the persisted turn; the streamed terminal event ends the wait.
      setStopping(false);
    }
  }

  /**
   * Approve or deny the action a turn paused for. The action is resolved by the
   * backend against the structured execution-context task id the turn recorded
   * (never by reading the message text), and the persisted transcript it returns
   * is the authority for what to display.
   */
  async function resolveConfirmation(message: ConversationMessage, approve: boolean) {
    const taskId = message.metadata?.task_id;
    if (!activeId || typeof taskId !== "string" || !taskId || resolvingConfirmation) return;
    setError(null);
    setResolvingConfirmation(taskId);
    try {
      const result = await api.resolveConfirmation(activeId, taskId, approve);
      setMessages(result.messages);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not resolve the confirmation");
    } finally {
      setResolvingConfirmation(null);
    }
  }

  /**
   * Edit a past message: the original wording is put back in the composer so it
   * can be changed and sent as a new turn. Nothing is silently rewritten in the
   * stored transcript — what Atlas actually said stays visible.
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
            <button type="button" onClick={() => setSearch("")} title="Clear search" aria-label="Clear search">
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

        {search.trim() && !hits.length && !searching ? (
          <p className="chat-empty-note">No conversations match “{search.trim()}”.</p>
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
                    aria-label="Conversation title"
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
                    type="button"
                    title="Rename conversation"
                    aria-label={`Rename ${conversation.title || "conversation"}`}
                    onClick={() => {
                      setRenamingId(conversation.id);
                      setRenameDraft(conversation.title || "");
                    }}
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    type="button"
                    title="Delete conversation"
                    aria-label={`Delete ${conversation.title || "conversation"}`}
                    onClick={() => deleteConversation(conversation.id)}
                  >
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

        {error ? (
          <div className="error-strip">
            <CircleAlert size={15} /> {error}
            <button onClick={() => setError(null)} aria-label="Dismiss error"><X size={14} /></button>
          </div>
        ) : null}

        <div className="chat-transcript" ref={transcriptRef}>
          {loadingConversation && !pending ? (
            <p className="chat-text chat-thinking">
              <Loader2 size={14} className="spin" /> Loading conversation…
            </p>
          ) : null}

          {messages.length === 0 && !pending && !loadingConversation ? (
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
              onResolveConfirmation={resolveConfirmation}
              resolvingConfirmation={resolvingConfirmation}
              canRegenerate={index === lastAssistantIndex && !pending}
            />
          ))}

          {pending ? (
            <div className="chat-message chat-message-assistant">
              <div className="chat-role">
                Atlas
                {pending.state !== "running" ? (
                  <em className={`chat-state chat-state-${pending.state}`}>{pending.state.replaceAll("_", " ")}</em>
                ) : null}
              </div>
              <div className="chat-bubble">
                {pending.tokens ? (
                  <p className="chat-text">{pending.tokens}</p>
                ) : pending.state === "waiting_for_confirmation" ? (
                  <p className="chat-text chat-thinking">
                    <LockKeyhole size={14} /> Waiting for your approval…
                  </p>
                ) : pending.state === "cancelled" ? (
                  <p className="chat-text chat-thinking">Stopped.</p>
                ) : pending.state === "failed" ? (
                  <p className="chat-text chat-thinking">
                    <CircleAlert size={14} /> The turn failed.
                  </p>
                ) : (
                  <p className="chat-text chat-thinking">
                    <Loader2 size={14} className="spin" /> {pending.started ? "Working…" : "Thinking…"}
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
            placeholder="Message Atlas…"
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
                  <button
                    type="button"
                    onClick={() => removeAttachment(attachment.path ?? "")}
                    title="Remove attachment"
                    aria-label={`Remove ${attachment.name}`}
                  >
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
                aria-label="Attachment file path"
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
            <span className="hint"><kbd>Enter</kbd> send · <kbd>Shift Enter</kbd> newline</span>
            {pending ? (
              <button
                type="button"
                className="button-muted"
                onClick={() => void stop()}
                disabled={pending.cancelled || stopping}
                aria-label="Stop the current turn"
              >
                {pending.cancelled || stopping ? (
                  <>
                    <Loader2 size={14} className="spin" /> Stopping…
                  </>
                ) : (
                  <>
                    <Square size={14} /> Stop
                  </>
                )}
              </button>
            ) : (
              <button type="submit" className="send-button" disabled={!draft.trim()} title="Send" aria-label="Send message">
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
  onResolveConfirmation,
  resolvingConfirmation,
  canRegenerate,
}: {
  message: ConversationMessage;
  onEdit: (message: ConversationMessage) => void;
  onRegenerate: (message: ConversationMessage) => void;
  onResolveConfirmation: (message: ConversationMessage, approve: boolean) => void;
  resolvingConfirmation: string | null;
  canRegenerate: boolean;
}) {
  const kind = (message.metadata?.response_kind || "") as ResponseKind | "";
  const activity = Array.isArray(message.metadata?.activity) ? message.metadata.activity : [];
  const cites = message.citations ?? [];
  const state = message.execution_state;
  const taskId = typeof message.metadata?.task_id === "string" ? message.metadata.task_id : null;
  const awaitingConfirmation = state === "waiting_for_confirmation" && Boolean(taskId);
  const resolving = awaitingConfirmation && resolvingConfirmation === taskId;
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
              {attachment.error ? <em> — {attachment.error}</em> : null}
            </span>
          ))}
        </div>
      ) : null}

      {awaitingConfirmation ? (
        <div className="approval-box">
          <div>
            <LockKeyhole size={18} />
            <div>
              <strong>Atlas is ready to act</strong>
              <span>
                This action changes your computer. Review it, then approve or reject to continue.
              </span>
            </div>
          </div>
          <div className="approval-actions">
            <button
              type="button"
              className="button-muted"
              onClick={() => onResolveConfirmation(message, false)}
              disabled={resolving}
            >
              <X size={15} /> Reject
            </button>
            <button
              type="button"
              className="button-primary"
              onClick={() => onResolveConfirmation(message, true)}
              disabled={resolving}
            >
              {resolving ? (
                <>
                  <Loader2 size={15} className="spin" /> Working…
                </>
              ) : (
                <>
                  <Check size={15} /> Confirm action
                </>
              )}
            </button>
          </div>
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
                {call.tool ?? "tool"} — {call.status ?? "unknown"}
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
  const [clearing, setClearing] = useState(false);
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

  /** Save an edited memory. A failed save is surfaced, never swallowed, so the
   *  editor stays open on a record that was not actually updated. */
  async function saveEdit(id: string) {
    try {
      await api.updateMemory(id, editDraft);
      setEditingId(null);
      await load(query);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not update that memory");
    }
  }

  async function forget(id: string) {
    try {
      await api.deleteMemory(id);
      await load(query);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not delete that memory");
    }
  }

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
            try {
              const next = await api.setMemoryEnabled(!status.enabled);
              setStatus({ enabled: next.enabled, count: next.count });
            } catch (reason) {
              setError(reason instanceof Error ? reason.message : "Could not change the memory setting");
            }
          }}
        >
          {status?.enabled ? "Disable memory" : "Enable memory"}
        </button>
        <button
          className="button-muted"
          type="button"
          disabled={clearing || records.length === 0}
          onClick={async () => {
            // Deleting all memory is irreversible, so it is confirmed first.
            if (!window.confirm("Delete every remembered item? This cannot be undone.")) return;
            setClearing(true);
            try {
              await api.clearMemory();
              await load(query);
            } catch (reason) {
              setError(reason instanceof Error ? reason.message : "Could not clear memory");
            } finally {
              setClearing(false);
            }
          }}
        >
          {clearing ? "Clearing…" : "Clear all"}
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
                    onKeyDown={(event) => {
                      if (event.key === "Enter") void saveEdit(record.id);
                      if (event.key === "Escape") setEditingId(null);
                    }}
                  />
                ) : (
                  <span>{record.text}</span>
                )}
              </div>
              <div className="memory-row-actions">
                {editingId === record.id ? (
                  <button type="button" onClick={() => void saveEdit(record.id)}>
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
                  type="button"
                  title="Forget"
                  onClick={() => void forget(record.id)}
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

