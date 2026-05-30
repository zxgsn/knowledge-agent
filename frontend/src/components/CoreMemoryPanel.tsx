import { useState, useEffect, useCallback, useRef } from "react";
import { Client } from "@langchain/langgraph-sdk";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Brain, ChevronRight, ChevronLeft, Loader2, Lock } from "lucide-react";

interface CoreMemoryBlock {
  label: string;
  value: string;
  description: string;
  limit: number;
  chars_current: number;
  read_only: boolean;
}

interface CoreMemoryPanelProps {
  threadId: string | null;
  refreshKey?: number;
}

const API_URL = import.meta.env.DEV
  ? "http://localhost:8123"
  : "http://localhost:8123";

const client = new Client({ apiUrl: API_URL });

const BLOCK_COLORS: Record<string, string> = {
  persona: "border-violet-500/40 bg-violet-500/5",
  human: "border-sky-500/40 bg-sky-500/5",
  knowledge_focus: "border-amber-500/40 bg-amber-500/5",
};

const BLOCK_BADGE_COLORS: Record<string, string> = {
  persona: "bg-violet-500/20 text-violet-300",
  human: "bg-sky-500/20 text-sky-300",
  knowledge_focus: "bg-amber-500/20 text-amber-300",
};

const DEFAULT_BLOCKS: CoreMemoryBlock[] = [
  { label: "persona", value: "", description: "The agent's personality and role definition.", limit: 5000, chars_current: 0, read_only: false },
  { label: "human", value: "", description: "Information about the user you are talking to.", limit: 5000, chars_current: 0, read_only: false },
  { label: "knowledge_focus", value: "", description: "Current research topics and areas of interest.", limit: 5000, chars_current: 0, read_only: false },
];

export function CoreMemoryPanel({ threadId, refreshKey }: CoreMemoryPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [blocks, setBlocks] = useState<CoreMemoryBlock[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [width, setWidth] = useState(320);
  const isResizing = useRef(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    const startX = e.clientX;
    const startWidth = width;

    const onMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;
      // Drag left = wider, drag right = narrower (panel is anchored to right)
      const newWidth = Math.max(250, Math.min(600, startWidth - (e.clientX - startX)));
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

  const fetchCoreMemory = useCallback(async () => {
    if (!threadId) return;
    setLoading(true);
    setError(null);
    try {
      const state = await client.threads.getState(threadId);
      const values = (state.values as any) || {};

      // core_memory can be {blocks: {...}, edit_history: [...]} or {label: value, ...}
      let raw: Record<string, string> = {};
      const cm = values.core_memory;
      if (cm && typeof cm === "object") {
        if (cm.blocks && typeof cm.blocks === "object") {
          // New format: {blocks: {label: value}, edit_history: [...]}
          raw = cm.blocks;
        } else {
          // Old format: {label: value}
          // Filter to only string values
          for (const [k, v] of Object.entries(cm)) {
            if (typeof v === "string") raw[k] = v;
          }
        }
      }

      const defaultMap = Object.fromEntries(DEFAULT_BLOCKS.map(b => [b.label, b]));
      const result: CoreMemoryBlock[] = [];

      for (const [label, value] of Object.entries(raw)) {
        const def = defaultMap[label];
        result.push({
          label,
          value,
          description: def?.description || "",
          limit: def?.limit || 5000,
          chars_current: value.length,
          read_only: def?.read_only || false,
        });
      }

      // Include default blocks not yet in state (so user always sees all blocks)
      for (const def of DEFAULT_BLOCKS) {
        if (!(def.label in raw)) {
          result.push({ ...def });
        }
      }

      setBlocks(result);
    } catch (e: any) {
      setError(e?.message || "Failed to load core memory");
      // Still show default blocks on error so the panel isn't blank
      setBlocks(DEFAULT_BLOCKS.map(b => ({ ...b })));
    } finally {
      setLoading(false);
    }
  }, [threadId]);

  useEffect(() => {
    if (isOpen && threadId) {
      fetchCoreMemory();
    }
  }, [isOpen, threadId, refreshKey, fetchCoreMemory]);

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed right-4 top-16 z-30 flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm transition-colors ${
          isOpen
            ? "bg-neutral-700 text-neutral-200"
            : "bg-neutral-800 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700"
        } border border-neutral-600/50 shadow-lg`}
        title="Core Memory"
      >
        <Brain className="h-4 w-4" />
        {isOpen ? (
          <ChevronRight className="h-3 w-3" />
        ) : (
          <ChevronLeft className="h-3 w-3" />
        )}
      </button>

      {/* Panel */}
      {isOpen && (
        <div
          className="fixed right-0 top-0 z-20 h-full border-l border-neutral-700 bg-neutral-900 shadow-2xl"
          style={{ width: `${width}px` }}
        >
          {/* Resize handle — left edge */}
          <div
            className="absolute top-0 left-0 w-1 h-full cursor-col-resize hover:bg-sky-500/50 transition-colors z-10"
            onMouseDown={handleMouseDown}
          />
          {/* Header */}
          <div className="flex items-center justify-between border-b border-neutral-700 px-4 py-3">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-violet-400" />
              <span className="text-sm font-medium text-neutral-200">
                Core Memory
              </span>
            </div>
            {threadId && (
              <button
                onClick={fetchCoreMemory}
                disabled={loading}
                className="text-xs text-neutral-400 hover:text-neutral-200 transition-colors disabled:opacity-50"
                title="Refresh"
              >
                {loading ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  "Refresh"
                )}
              </button>
            )}
          </div>

          {/* Content */}
          <ScrollArea className="h-[calc(100%-49px)]">
            <div className="p-4 space-y-3">
              {!threadId ? (
                <p className="text-sm text-neutral-500 text-center py-8">
                  Select a conversation to view core memory.
                </p>
              ) : loading && blocks.length === 0 ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-5 w-5 text-neutral-400 animate-spin" />
                </div>
              ) : error ? (
                <p className="text-sm text-red-400 text-center py-8">{error}</p>
              ) : blocks.length === 0 ? (
                <p className="text-sm text-neutral-500 text-center py-8">
                  Core memory is empty.
                </p>
              ) : (
                blocks.map((block) => (
                  <BlockCard key={block.label} block={block} />
                ))
              )}
            </div>
          </ScrollArea>
        </div>
      )}
    </>
  );
}

function BlockCard({ block }: { block: CoreMemoryBlock }) {
  const [expanded, setExpanded] = useState(false);
  const usagePercent = Math.round((block.chars_current / block.limit) * 100);
  const borderColor = BLOCK_COLORS[block.label] || "border-neutral-600/40 bg-neutral-800/50";
  const badgeColor = BLOCK_BADGE_COLORS[block.label] || "bg-neutral-600/20 text-neutral-300";

  return (
    <div
      className={`rounded-lg border p-3 cursor-pointer transition-colors hover:brightness-110 ${borderColor}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className={`text-xs px-2 py-0.5 ${badgeColor}`}>
            {block.label}
          </Badge>
          {block.read_only && (
            <Lock className="h-3 w-3 text-neutral-500" />
          )}
        </div>
        <span className="text-xs text-neutral-500">
          {block.chars_current}/{block.limit}
        </span>
      </div>

      {block.description && (
        <p className="text-xs text-neutral-500 mb-2">{block.description}</p>
      )}

      <div className="w-full h-1 rounded-full bg-neutral-700 mb-2">
        <div
          className={`h-full rounded-full transition-all ${
            usagePercent >= 80
              ? "bg-red-400"
              : usagePercent >= 50
                ? "bg-amber-400"
                : "bg-emerald-400"
          }`}
          style={{ width: `${Math.min(usagePercent, 100)}%` }}
        />
      </div>

      {block.value ? (
        <p
          className={`text-xs text-neutral-300 whitespace-pre-wrap ${
            expanded ? "" : "line-clamp-3"
          }`}
        >
          {block.value}
        </p>
      ) : (
        <p className="text-xs text-neutral-600 italic">Empty</p>
      )}

      {block.value && block.value.length > 150 && (
        <p className="text-xs text-neutral-600 mt-1">
          {expanded ? "Click to collapse" : "Click to expand"}
        </p>
      )}
    </div>
  );
}
