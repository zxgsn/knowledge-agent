import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, Plus, PanelLeftClose, PanelLeft } from "lucide-react";
import { Client } from "@langchain/langgraph-sdk";

const API_URL = import.meta.env.DEV
  ? "http://localhost:2024"
  : "http://localhost:8123";

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

  const fetchThreads = useCallback(async () => {
    setLoading(true);
    try {
      const client = new Client({ apiUrl: API_URL });
      const threadList = await client.threads.search({
        limit: 50,
        sortBy: "created_at",
        sortOrder: "desc",
      });

      const parsed: Thread[] = threadList.map((t: any) => {
        const msgs = t.values?.messages || [];
        const firstHuman = msgs.find((m: any) => m.type === "human");
        return {
          thread_id: t.thread_id,
          created_at: t.created_at || t.metadata?.created_at || "",
          first_message: firstHuman?.content || "",
        };
      });
      setThreads(parsed);
    } catch (err) {
      console.error("Failed to fetch threads:", err);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (isOpen) fetchThreads();
  }, [isOpen, fetchThreads]);

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
    <div className="flex flex-col w-64 min-w-[256px] bg-neutral-900 border-r border-neutral-700 h-full">
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
      <ScrollArea className="flex-1">
        {loading ? (
          <div className="text-center text-neutral-500 py-8 text-sm">Loading...</div>
        ) : threads.length === 0 ? (
          <div className="text-center text-neutral-500 py-8 text-sm">No conversations yet.</div>
        ) : (
          <div className="p-2 space-y-1">
            {threads.map((t) => (
              <button
                key={t.thread_id}
                onClick={() => onThreadSelect(t.thread_id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  currentThreadId === t.thread_id
                    ? "bg-neutral-700 text-neutral-100"
                    : "text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                }`}
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                  <span className="truncate flex-1">
                    {t.first_message ? truncate(t.first_message, 40) : "New conversation"}
                  </span>
                </div>
                {t.created_at && (
                  <span className="text-xs text-neutral-600 ml-5">
                    {timeAgo(t.created_at)}
                  </span>
                )}
              </button>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
