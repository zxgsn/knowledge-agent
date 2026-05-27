import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { Routes, Route } from "react-router-dom";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { DocumentLibrary } from "@/components/DocumentLibrary";
import { MemoryAnalytics } from "@/components/MemoryAnalytics";
import { CoreMemoryPanel } from "@/components/CoreMemoryPanel";
import { ThreadSidebar } from "@/components/ThreadSidebar";
import { Button } from "@/components/ui/button";

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
  const thread = useStream<{
    messages: Message[];
    core_memory: Record<string, string>;
    mode: string;
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "agent",
    messagesKey: "messages",
    threadId: currentThreadId ?? undefined,
    onCustomEvent: (event: any) => {
      const payload = event?.name ? event : event?.data ? event.data : event;
      if (payload?.name === "progress") {
        const { stage, detail } = payload.data || {};
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
      if (msg.includes("CancelledError") || msg.includes("User interrupted")) {
        return;
      }
      setError(msg);
    },
  });

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
      setCoreMemoryRefreshKey((k) => k + 1);
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  // Clear the submission lock when loading finishes (handles the race where
  // the first stream's finally block fires after the second stream starts).
  useEffect(() => {
    if (!thread.isLoading) {
      submitLockRef.current = false;
    }
  }, [thread.isLoading]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, mode: string) => {
      if (!submittedInputValue.trim()) return;
      if (submitLockRef.current) return;

      // Cancel any in-progress run before submitting a new one
      if (thread.isLoading) {
        thread.stop();
        // Wait one tick for the aborted stream's finally block to execute
        // before starting a new stream, preventing the stale finally from
        // corrupting the new stream's shared state.
        setTimeout(() => {
          submitLockRef.current = true;
          setProcessedEventsTimeline([]);
          setError(null);
          hasFinalizeEventOccurredRef.current = false;

          const newMessages: Message[] = [
            ...(thread.messages || []),
            {
              type: "human",
              content: submittedInputValue,
              id: Date.now().toString(),
            },
          ];
          thread.submit({
            messages: newMessages,
            mode: mode,
          });
        }, 0);
        return;
      }

      submitLockRef.current = true;
      setProcessedEventsTimeline([]);
      setError(null);
      hasFinalizeEventOccurredRef.current = false;

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      thread.submit({
        messages: newMessages,
        mode: mode,
      });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
  }, [thread]);

  const handleThreadSelect = useCallback((threadId: string | null) => {
    setCurrentThreadId(threadId);
    setProcessedEventsTimeline([]);
    setHistoricalActivities({});
    setError(null);
  }, []);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <Routes>
        <Route path="/library" element={<DocumentLibrary />} />
        <Route path="/analytics" element={<MemoryAnalytics />} />
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
            <main className="h-full flex-1 max-w-4xl mx-auto">
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
              ) : thread.messages.length === 0 ? (
                <WelcomeScreen
                  handleSubmit={handleSubmit}
                  isLoading={thread.isLoading}
                  onCancel={handleCancel}
                />
              ) : (
                <ChatMessagesView
                  messages={thread.messages}
                  isLoading={thread.isLoading}
                  scrollAreaRef={scrollAreaRef}
                  onSubmit={handleSubmit}
                  onCancel={handleCancel}
                  liveActivityEvents={processedEventsTimeline}
                  historicalActivities={historicalActivities}
                />
              )}
            </main>
            </div>
          }
        />
      </Routes>
    </div>
  );
}
