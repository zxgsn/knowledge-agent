import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { MessageSquare, Plus, PanelLeftClose, PanelLeft, Trash2, Loader2 } from "lucide-react";
import { Client } from "@langchain/langgraph-sdk";

const API_URL = import.meta.env.VITE_LANGGRAPH_URL || "http://localhost:2024";

interface Thread {
  thread_id: string;
  created_at: string;
  first_message?: string;
}

interface ThreadSidebarProps {
  currentThreadId: string | null;
  onThreadSelect: (threadId: string | null) => void;
  isOpen: boolean;
  onToggle: () => void;
}

function timeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return date.toLocaleDateString();
}

function truncate(text: string, maxLen: number): string {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "...";
}

export function ThreadSidebar({
  currentThreadId,
  onThreadSelect,
  isOpen,
  onToggle,
}: ThreadSidebarProps) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [width, setWidth] = useState(256);
  const isResizing = useRef(false);
  const hasLoaded = useRef(false);

  const handleDelete = useCallback(async (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Delete this conversation and its recall memory?")) return;
    setDeletingId(threadId);
    try {
      const client = new Client({ apiUrl: API_URL });
      await client.threads.delete(threadId);
      try {
        const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";
        await fetch(`${API_BASE}/api/recall/entries?thread_id=${threadId}`, { method: "DELETE" });
      } catch { /* best-effort */ }
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId));
      if (currentThreadId === threadId) onThreadSelect(null);
    } catch (err) {
      console.error("Failed to delete thread:", err);
    } finally {
      setDeletingId(null);
    }
  }, [currentThreadId, onThreadSelect]);

  const fetchThreads = useCallback(async () => {
    setLoading(true);
    try {
      const client = new Client({ apiUrl: API_URL });
      const threadList = await client.threads.search({
        limit: 50,
        sortBy: "created_at",
        sortOrder: "desc",
      });

      // Build base thread list from search metadata (instant, no extra API calls)
      const baseThreads: Thread[] = (threadList as any[]).map((t) => ({
        thread_id: t.thread_id,
        created_at: t.created_at || t.metadata?.created_at || "",
        first_message: "",
      }));

      // Show threads immediately with empty previews
      baseThreads.sort((a, b) => (b.created_at > a.created_at ? 1 : -1));
      setThreads(baseThreads);

      // Then fetch first message previews in parallel (best-effort)
      const BATCH_SIZE = 5;
      for (let i = 0; i < baseThreads.length; i += BATCH_SIZE) {
        const batch = baseThreads.slice(i, i + BATCH_SIZE);
        const results = await Promise.allSettled(
          batch.map(async (t) => {
            const hist = await client.threads.getHistory(t.thread_id, { limit: 1 });
            const state = hist?.[0]?.values as Record<string, any> | undefined;
            const msgs = state?.messages;
            if (Array.isArray(msgs)) {
              const firstHuman = msgs.find((m: any) => m.type === "human");
              return { threadId: t.thread_id, preview: firstHuman?.content || "" };
            }
            return null;
          })
        );
        // Update threads with fetched previews
        setThreads((prev) => {
          const updated = [...prev];
          for (const r of results) {
            if (r.status === "fulfilled" && r.value) {
              const idx = updated.findIndex((t) => t.thread_id === r.value!.threadId);
              if (idx >= 0) updated[idx] = { ...updated[idx], first_message: r.value!.preview };
            }
          }
          return updated;
        });
      }

      hasLoaded.current = true;
    } catch (err) {
      console.error("Failed to fetch threads:", err);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (isOpen && !hasLoaded.current) fetchThreads();
  }, [isOpen, fetchThreads]);

  // Refresh list when a new thread is created (currentThreadId becomes non-null and not in list)
  useEffect(() => {
    if (currentThreadId && !threads.find((t) => t.thread_id === currentThreadId)) {
      hasLoaded.current = false;
      if (isOpen) fetchThreads();
    }
  }, [currentThreadId, threads, isOpen, fetchThreads]);

  // Resize handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    const startX = e.clientX;
    const startWidth = width;

    const onMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;
      const newWidth = Math.max(200, Math.min(500, startWidth + (e.clientX - startX)));
      setWidth(newWidth);
    };

    const onMouseUp = () => {
      isResizing.current = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };

    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, [width]);

  if (!isOpen) {
    return (
      <div className="flex flex-col items-center pt-4 px-1 gap-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className="text-neutral-400 hover:text-neutral-100"
          title="Open sidebar"
        >
          <PanelLeft className="w-5 h-5" />
        </Button>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col bg-neutral-900 border-r border-neutral-700 h-full relative"
      style={{ width: `${width}px`, minWidth: `${width}px` }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-neutral-700">
        <span className="text-sm font-medium text-neutral-300">Conversations</span>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onThreadSelect(null)}
            className="text-neutral-400 hover:text-neutral-100 h-7 w-7"
            title="New chat"
          >
            <Plus className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="text-neutral-400 hover:text-neutral-100 h-7 w-7"
            title="Close sidebar"
          >
            <PanelLeftClose className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="text-center text-neutral-500 py-8 text-sm">Loading...</div>
        ) : threads.length === 0 ? (
          <div className="text-center text-neutral-500 py-8 text-sm">No conversations yet.</div>
        ) : (
          <div className="p-2 space-y-1">
            {threads.map((t) => (
              <div
                key={t.thread_id}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
                  currentThreadId === t.thread_id
                    ? "bg-neutral-700 text-neutral-100"
                    : "text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                }`}
                onClick={() => onThreadSelect(t.thread_id)}
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                  <span className="truncate flex-1 min-w-0">
                    {t.first_message ? truncate(t.first_message, 30) : "New conversation"}
                  </span>
                  <button
                    onClick={(e) => handleDelete(t.thread_id, e)}
                    disabled={deletingId === t.thread_id}
                    className="flex-shrink-0 p-1 rounded hover:bg-red-500/20 text-neutral-500 hover:text-red-400 transition-colors"
                    title="Delete conversation"
                  >
                    {deletingId === t.thread_id ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Trash2 className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>
                {t.created_at && (
                  <span className="text-xs text-neutral-600 ml-5">
                    {timeAgo(t.created_at)}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Resize handle */}
      <div
        className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-sky-500/50 transition-colors z-10"
        onMouseDown={handleMouseDown}
      />
    </div>
  );
}
