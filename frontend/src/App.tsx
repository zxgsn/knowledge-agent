import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
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
    onUpdateEvent: (event: any) => {
      let processedEvent: ProcessedEvent | null = null;
      if (event.route_intent) {
        processedEvent = {
          title: "Understanding Intent",
          data: `Mode: ${event.route_intent?.mode || "chat"}`,
        };
      } else if (event.generate_query) {
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
        const resultCount = recallOp?.result_count ?? 0;
        const archivalResults = event.recall_memory?.archival_results ?? [];
        const previews = archivalResults
          .slice(0, 3)
          .map((r: any) => {
            const content = r.content || "";
            const preview = content.length > 60 ? content.slice(0, 60) + "..." : content;
            return `[${r.source || "archival"}] ${preview} (${(r.score ?? 0).toFixed(2)})`;
          });
        processedEvent = {
          title: resultCount > 0 ? `Memory Retrieved (${resultCount} results)` : "Memory Search (no results)",
          data: previews.length > 0 ? previews.join("\n") : "No relevant memories found.",
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
      } else if (event.save_to_archival) {
        processedEvent = {
          title: "Saving to Memory",
          data: "Storing findings in long-term memory...",
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
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, mode: string) => {
      if (!submittedInputValue.trim()) return;

      // Cancel any in-progress run before submitting a new one
      if (thread.isLoading) {
        thread.stop();
      }

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

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="h-full w-full max-w-4xl mx-auto">
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
  );
}
