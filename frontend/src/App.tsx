import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { Routes, Route } from "react-router-dom";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { DocumentLibrary } from "@/components/DocumentLibrary";
import { MemoryAnalytics } from "@/components/MemoryAnalytics";
import { MemorySearchPanel } from "@/components/MemorySearchPanel";
import { KnowledgeGraphView } from "@/components/KnowledgeGraphView";
import { CoreMemoryPanel } from "@/components/CoreMemoryPanel";
import { ThreadSidebar } from "@/components/ThreadSidebar";
import { Button } from "@/components/ui/button";
import { NavLink } from "react-router-dom";
import { LANGGRAPH_API_URL } from "@/lib/api";
import { cn } from "@/lib/utils";
import { MessageSquare, BookOpen, BarChart3, Search, Network } from "lucide-react";

const DEBUG_SERVER_URL =
  import.meta.env.VITE_DEBUG_SERVER_URL || "http://127.0.0.1:7777/event";
const DEBUG_SESSION_ID = "frontend-network-error";
const DEBUG_RUN_ID = "pre-fix";

// #region debug-point A:frontend-stream-lifecycle
function reportDebug(
  hypothesisId: string,
  location: string,
  msg: string,
  data: Record<string, unknown> = {}
) {
  fetch(DEBUG_SERVER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionId: DEBUG_SESSION_ID,
      runId: DEBUG_RUN_ID,
      hypothesisId,
      location,
      msg,
      data,
      ts: Date.now(),
    }),
  }).catch(() => {});
}
// #endregion

export default function App() {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [coreMemoryRefreshKey, setCoreMemoryRefreshKey] = useState(0);
  const submitLockRef = useRef(false);
  const pendingMsgRef = useRef<{ value: string; mode: string } | null>(null);
  const currentRunModeRef = useRef<string>("chat");
  const [pendingMessageText, setPendingMessageText] = useState<string | null>(null);
  // Persisted message list — useStream's thread.messages can become empty during
  // stream transitions (new stream starts before backend responds), which would
  // flash the WelcomeScreen. We maintain our own copy that only updates when the
  // hook returns a non-empty array.
  const [displayMessages, setDisplayMessages] = useState<Message[]>([]);
  const prevThreadIdRef = useRef<string | null>(null);
  const thread = useStream<{
    messages: Message[];
    core_memory: Record<string, string>;
    mode: string;
  }>({
    apiUrl: LANGGRAPH_API_URL,
    assistantId: "agent",
    messagesKey: "messages",
    threadId: currentThreadId ?? undefined,
    onCustomEvent: (event: any) => {
      const payload = event?.name ? event : event?.data ? event.data : event;
      if (payload?.name === "progress") {
        const { stage, detail } = payload.data || {};
        // #region debug-point D:progress-events
        reportDebug("D", "frontend/src/App.tsx:onCustomEvent", "[DEBUG] progress event received", {
          stage: stage || "",
          detail: detail || "",
          threadId: currentThreadId ?? "new",
        });
        // #endregion
        const titleMap: Record<string, string> = {
          memory_search: "Searching Memory",
          evaluate_recall: "Evaluating Recall",
          generate_query: "Generating Queries",
          web_research: "Web Research",
          reflection: "Reflection",
          save_to_archival: "Saving to Memory",
          ingest_document: "Ingesting Document",
          memory_pipeline: "Memory Pipeline",
          consolidate_memory: "Consolidating Memory",
          respond: "Generating Response",
        };
        const title = titleMap[stage] || stage || "Processing";
        setProcessedEventsTimeline((prev) => {
          // Remove placeholder and add real progress event
          const filtered = prev.filter((e) => e.data !== "__pending__");
          const last = filtered[filtered.length - 1];
          if (last && last.title === title && last.data === detail) return filtered;
          return [...filtered, { title, data: detail || "" }];
        });
      }
    },
    onUpdateEvent: (event: any) => {
      // Remove any pending placeholders when real events arrive
      const removePending = () => {
        setProcessedEventsTimeline((prev) =>
          prev.filter((e) => e.data !== "__pending__")
        );
      };

      let processedEvent: ProcessedEvent | null = null;
      if (event.route_intent) {
        processedEvent = {
          title: "Understanding Intent",
          data: `Mode: ${event.route_intent?.mode || "chat"}`,
        };
        // Add placeholder for next step
        setProcessedEventsTimeline((prev) => [
          ...prev,
          processedEvent!,
          { title: "Searching Memory", data: "__pending__" },
        ]);
        return;
      } else if (event.recall_memory || event.evaluate_recall || event.generate_query) {
        removePending();
      }
      if (event.generate_query) {
        processedEvent = {
          title: "Generating Search Queries",
          data: event.generate_query?.search_query?.join(", ") || "",
        };
      } else if (event.web_research) {
        processedEvent = {
          title: "Web Research",
          data: "Searching and summarizing results...",
        };
      } else if (event.reflection) {
        processedEvent = {
          title: "Reflection",
          data: event.reflection?.knowledge_gap || "Analyzing research results",
        };
      } else if (event.recall_memory) {
        const memOps = event.recall_memory?.memory_operations;
        const recallOp = Array.isArray(memOps) ? memOps.find((op: any) => op.type === "recall") : null;
        const archivalCount = recallOp?.archival_count ?? 0;
        const recallCount = recallOp?.recall_count ?? 0;
        const totalResults = archivalCount + recallCount;
        const archivalResults = event.recall_memory?.archival_results ?? [];
        const recallResults = event.recall_memory?.recall_results ?? [];
        const memoryItems = [
          ...archivalResults.slice(0, 3).map((r: any) => ({
            type: "archival",
            content: r.content || "",
            source: r.source || "",
            sourceType: r.source_type || "",
            document: r.document || "",
            namespace: r.namespace || "",
            timestamp: r.timestamp || "",
            score: r.score ?? 0,
          })),
          ...recallResults.slice(0, 2).map((r: any) => ({
            type: "recall",
            content: r.content || "",
            role: r.role || "",
            threadId: r.thread_id || "",
            score: r.score ?? 0,
          })),
        ];
        processedEvent = {
          title: totalResults > 0
            ? `Memory Retrieved (${archivalCount} archival, ${recallCount} recall)`
            : "Memory Search (no results)",
          data: memoryItems.length > 0 ? memoryItems : "No relevant memories found.",
        };
      } else if (event.evaluate_recall) {
        const memOps = event.evaluate_recall?.memory_operations;
        const evalOp = Array.isArray(memOps) ? memOps.find((op: any) => op.type === "evaluate") : null;
        const isSufficient = evalOp?.is_sufficient ?? false;
        const reason = evalOp?.reason || "Evaluating...";
        processedEvent = {
          title: isSufficient ? "Memory Sufficient" : "Memory Insufficient — Searching Web",
          data: reason,
        };
        // Add placeholder for next step
        if (!isSufficient) {
          setProcessedEventsTimeline((prev) => [
            ...prev,
            processedEvent!,
            { title: "Generating Queries", data: "__pending__" },
          ]);
          return;
        }
      } else if (event.memory_pipeline) {
        const memOps = event.memory_pipeline?.memory_operations;
        const extractOp = Array.isArray(memOps) ? memOps.find((op: any) => op.type === "pipeline_extract") : null;
        const judgmentOp = Array.isArray(memOps) ? memOps.find((op: any) => op.type === "pipeline_judgment") : null;
        if (extractOp) {
          const extracted = extractOp.facts_count ?? 0;
          const stored = extractOp.stored ?? 0;
          const updated = extractOp.updated ?? 0;
          const deleted = extractOp.deleted ?? 0;
          const skipped = extractOp.skipped ?? 0;
          const existing = extractOp.existing_memories ?? 0;
          if (extracted > 0) {
            const parts = [];
            if (stored > 0) parts.push(`${stored} added`);
            if (updated > 0) parts.push(`${updated} updated`);
            if (deleted > 0) parts.push(`${deleted} deleted`);
            if (skipped > 0) parts.push(`${skipped} unchanged`);
            processedEvent = {
              title: "Memory Pipeline",
              data: `${extracted} operations (${existing} existing): ${parts.join(", ")}`,
            };
          } else {
            processedEvent = {
              title: "Memory Pipeline",
              data: "No new facts extracted.",
            };
          }
        } else if (judgmentOp) {
          const memorable = judgmentOp.memorable ?? false;
          processedEvent = {
            title: "Memory Judgment",
            data: memorable ? "Content marked as memorable." : "Not memorable — skipping.",
          };
        } else {
          processedEvent = {
            title: "Memory Pipeline",
            data: "Checked — no new memories.",
          };
        }
      } else if (event.save_to_archival) {
        const memOps = event.save_to_archival?.memory_operations;
        const saveOp = Array.isArray(memOps) ? memOps.find((op: any) => op.type === "save") : null;
        const topic = saveOp?.topic || "";
        const preview = saveOp?.content_preview || "";
        processedEvent = {
          title: "Saved to Archival Memory",
          data: topic ? `Topic: ${topic}\n${preview}` : preview || "Storing findings in long-term memory...",
        };
      } else if (event.consolidate_memory) {
        const memOps = event.consolidate_memory?.memory_operations;
        const parts: string[] = [];
        if (Array.isArray(memOps)) {
          for (const op of memOps) {
            if (op.type === "consolidate") {
              const ns = op.namespace || "?";
              const merged = op.merged ?? 0;
              const deleted = op.deleted ?? 0;
              if (merged > 0 || deleted > 0) {
                parts.push(`[${ns}] ${merged} merged, ${deleted} removed`);
              }
            } else if (op.type === "cleanup") {
              const ns = op.namespace || "?";
              const expired = op.expired_deleted ?? 0;
              const excess = op.excess_deleted ?? 0;
              if (expired > 0 || excess > 0) {
                parts.push(`[${ns}] ${expired} expired, ${excess} excess`);
              }
            }
          }
        }
        if (parts.length > 0) {
          processedEvent = {
            title: "Memory Consolidation",
            data: parts.join("; "),
          };
        } else {
          processedEvent = {
            title: "Memory Consolidation",
            data: "Checked — memory is clean.",
          };
        }
      } else if (event.ingest_document) {
        const ingestResult = event.ingest_document?.ingest_result;
        processedEvent = {
          title: "Document Ingested",
          data: ingestResult || "Document stored in archival memory.",
        };
      } else if (event.respond) {
        processedEvent = {
          title: "Generating Response",
          data: "Composing the final answer.",
        };
        hasFinalizeEventOccurredRef.current = true;
      } else if (event.memory_operation) {
        const op = event.memory_operation;
        let title = "Memory Operation";
        let data = "Processing memory...";
        if (op.type === "recall") {
          title = "Searching Memory";
          data = `Found ${op.result_count} results for: ${op.query}`;
        } else if (op.type === "save") {
          title = "Saving to Memory";
          data = `Stored: ${op.topic || op.content_preview}`;
        } else if (op.type === "ingest") {
          title = "Ingesting Document";
          data = `${op.document}: ${op.chunks_count} chunks stored`;
        }
        processedEvent = { title, data };
      }
      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
    onError: (error: any) => {
      // Ignore cancellation errors — these happen when a user interrupts a running stream
      const msg =
        error?.message ?? error?.error ?? (typeof error === "string" ? error : "An error occurred");
      // #region debug-point A:on-error
      reportDebug("A", "frontend/src/App.tsx:onError", "[DEBUG] stream error surfaced to UI", {
        message: msg,
        rawError:
          error instanceof Error
            ? { name: error.name, message: error.message, stack: error.stack }
            : error,
        threadId: currentThreadId ?? "new",
        hasMessages: displayMessages.length > 0,
      });
      // #endregion
      if (msg.includes("CancelledError") || msg.includes("User interrupted")) {
        return;
      }
      setError(msg);
    },
  });

  // Sync displayMessages: only update when useStream returns non-empty messages.
  // This prevents the WelcomeScreen flash when useStream briefly returns []
  // during a stream transition.
  useEffect(() => {
    if (thread.messages.length > 0) {
      setDisplayMessages(thread.messages);
    }
  }, [thread.messages]);

  // Reset displayMessages when switching to a different thread
  useEffect(() => {
    if (currentThreadId !== prevThreadIdRef.current) {
      prevThreadIdRef.current = currentThreadId;
      setDisplayMessages([]);
    }
  }, [currentThreadId]);

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [displayMessages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      displayMessages.length > 0
    ) {
      const lastMessage = displayMessages[displayMessages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
      setCoreMemoryRefreshKey((k) => k + 1);
    }
  }, [displayMessages, thread.isLoading, processedEventsTimeline]);

  // Clear the submission lock when loading finishes (handles the race where
  // the first stream's finally block fires after the second stream starts).
  // Also add a safety timeout: if isLoading stays true for too long, force-clear.
  useEffect(() => {
    if (!thread.isLoading) {
      submitLockRef.current = false;
    }
  }, [thread.isLoading]);

  // Safety net: if isLoading is true for more than 5 minutes, clear the lock
  // and surface a timeout error. This prevents permanent UI freezes.
  useEffect(() => {
    if (!thread.isLoading) return;
    const isResearchMode = currentRunModeRef.current === "research";
    const timeoutMs = isResearchMode ? 15 * 60 * 1000 : 5 * 60 * 1000;
    const timeout = setTimeout(() => {
      // #region debug-point D:safety-timeout
      reportDebug("D", "frontend/src/App.tsx:safety-timeout", "[DEBUG] frontend safety timeout fired", {
        threadId: currentThreadId ?? "new",
        displayMessageCount: displayMessages.length,
        mode: currentRunModeRef.current,
        timeoutMs,
      });
      // #endregion
      thread.stop();
      submitLockRef.current = false;
      setError(
        isResearchMode
          ? "Research request timed out after 15 minutes. The backend may still be processing or responding too slowly."
          : "Request timed out. The backend may be unreachable or processing slowly."
      );
    }, timeoutMs);
    return () => clearTimeout(timeout);
  }, [thread.isLoading, thread, currentThreadId, displayMessages.length]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, mode: string) => {
      if (!submittedInputValue.trim()) return;
      if (submitLockRef.current) return;

      // If a stream is in progress, queue the message instead of interrupting.
      // Interrupting via thread.stop() + immediate thread.submit() can cause
      // useStream to lose accumulated messages, resetting the page to blank.
      if (thread.isLoading) {
        pendingMsgRef.current = { value: submittedInputValue, mode };
        setPendingMessageText(submittedInputValue);
        return;
      }

      submitLockRef.current = true;
      currentRunModeRef.current = mode;
      setProcessedEventsTimeline([]);
      setError(null);
      hasFinalizeEventOccurredRef.current = false;
      // #region debug-point E:submit
      reportDebug("E", "frontend/src/App.tsx:handleSubmit", "[DEBUG] submit stream request", {
        mode,
        threadId: currentThreadId ?? "new",
        messageLength: submittedInputValue.length,
        displayMessageCount: displayMessages.length,
        apiUrl: LANGGRAPH_API_URL,
      });
      // #endregion

      const newMessages: Message[] = [
        ...displayMessages,
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      thread.submit(
        { messages: newMessages, mode: mode },
        {
          optimisticValues: (prev) => ({
            ...prev,
            messages: newMessages,
          }),
        }
      );
    },
    [thread, displayMessages]
  );

  // Submit queued message once the current stream finishes
  useEffect(() => {
    if (!thread.isLoading && pendingMsgRef.current && !submitLockRef.current) {
      const { value, mode } = pendingMsgRef.current;
      pendingMsgRef.current = null;
      setPendingMessageText(null);

      submitLockRef.current = true;
      currentRunModeRef.current = mode;
      setProcessedEventsTimeline([]);
      setError(null);
      hasFinalizeEventOccurredRef.current = false;

      const newMessages: Message[] = [
        ...displayMessages,
        { type: "human", content: value, id: Date.now().toString() },
      ];
      thread.submit(
        { messages: newMessages, mode },
        {
          optimisticValues: (prev) => ({
            ...prev,
            messages: newMessages,
          }),
        }
      );
    }
  }, [thread.isLoading, displayMessages, thread.submit]);

  const handleCancel = useCallback(() => {
    pendingMsgRef.current = null;
    setPendingMessageText(null);
    thread.stop();
  }, [thread]);

  const handleThreadSelect = useCallback((threadId: string | null) => {
    setCurrentThreadId(threadId);
    setProcessedEventsTimeline([]);
    setHistoricalActivities({});
    setError(null);
  }, []);

  return (
    <div className="flex flex-col h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      {/* Top navigation bar */}
      <nav className="flex items-center gap-1 px-4 py-2 bg-neutral-900 border-b border-neutral-700 shrink-0">
        <NavLink
          to="/"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
              isActive
                ? "bg-neutral-700 text-neutral-100"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800"
            )
          }
        >
          <MessageSquare className="w-3.5 h-3.5" />
          Chat
        </NavLink>
        <NavLink
          to="/library"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
              isActive
                ? "bg-neutral-700 text-neutral-100"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800"
            )
          }
        >
          <BookOpen className="w-3.5 h-3.5" />
          Library
        </NavLink>
        <NavLink
          to="/analytics"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
              isActive
                ? "bg-neutral-700 text-neutral-100"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800"
            )
          }
        >
          <BarChart3 className="w-3.5 h-3.5" />
          Analytics
        </NavLink>
        <NavLink
          to="/memory-search"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
              isActive
                ? "bg-neutral-700 text-neutral-100"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800"
            )
          }
        >
          <Search className="w-3.5 h-3.5" />
          Search
        </NavLink>
        <NavLink
          to="/knowledge-graph"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
              isActive
                ? "bg-neutral-700 text-neutral-100"
                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800"
            )
          }
        >
          <Network className="w-3.5 h-3.5" />
          Graph
        </NavLink>
      </nav>

      <Routes>
        <Route path="/library" element={<DocumentLibrary />} />
        <Route path="/analytics" element={<MemoryAnalytics />} />
        <Route path="/memory-search" element={<MemorySearchPanel />} />
        <Route path="/knowledge-graph" element={<KnowledgeGraphView />} />
        <Route
          path="*"
          element={
            <div className="flex h-full w-full">
              <ThreadSidebar
                currentThreadId={currentThreadId}
                onThreadSelect={handleThreadSelect}
                isOpen={sidebarOpen}
                onToggle={() => setSidebarOpen(!sidebarOpen)}
              />
              <CoreMemoryPanel
                threadId={currentThreadId}
                refreshKey={coreMemoryRefreshKey}
              />
            <main className="h-full flex-1 max-w-4xl mx-auto overflow-hidden">
              {error ? (
                <div className="flex flex-col items-center justify-center h-full">
                  <div className="flex flex-col items-center justify-center gap-4">
                    <h1 className="text-2xl text-red-400 font-bold">Error</h1>
                    <p className="text-red-400">{JSON.stringify(error)}</p>
                    <Button
                      variant="destructive"
                      onClick={() => window.location.reload()}
                    >
                      Retry
                    </Button>
                  </div>
                </div>
              ) : displayMessages.length === 0 ? (
                <WelcomeScreen
                  handleSubmit={handleSubmit}
                  isLoading={thread.isLoading}
                  onCancel={handleCancel}
                />
              ) : (
                <>
                  <ChatMessagesView
                    messages={displayMessages}
                    isLoading={thread.isLoading}
                    scrollAreaRef={scrollAreaRef}
                    onSubmit={handleSubmit}
                    onCancel={handleCancel}
                    liveActivityEvents={processedEventsTimeline}
                    historicalActivities={historicalActivities}
                  />
                  {pendingMessageText && (
                    <div className="mx-auto max-w-4xl px-4 pb-1">
                      <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-900/30 border border-amber-700/50 rounded-lg text-xs text-amber-300">
                        <span className="inline-block w-1.5 h-1.5 bg-amber-400 rounded-full animate-pulse" />
                        <span className="truncate">Queued: {pendingMessageText}</span>
                      </div>
                    </div>
                  )}
                </>
              )}
            </main>
            </div>
          }
        />
      </Routes>
    </div>
  );
}
