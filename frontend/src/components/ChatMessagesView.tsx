import type React from "react";
import type { Message } from "@langchain/langgraph-sdk";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, Copy, CopyCheck, BookMarked, BookOpen, ChevronDown, ChevronUp, BarChart3 } from "lucide-react";
import { InputForm } from "@/components/InputForm";
import { Button } from "@/components/ui/button";
import { useState, ReactNode } from "react";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  ActivityTimeline,
  ProcessedEvent,
} from "@/components/ActivityTimeline"; // Assuming ActivityTimeline is in the same dir or adjust path

// Markdown component props type from former ReportView
type MdComponentProps = {
  className?: string;
  children?: ReactNode;
  [key: string]: any;
};

// Markdown components (from former ReportView.tsx)
const mdComponents = {
  h1: ({ className, children, ...props }: MdComponentProps) => (
    <h1 className={cn("text-2xl font-bold mt-4 mb-2", className)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: MdComponentProps) => (
    <h2 className={cn("text-xl font-bold mt-3 mb-2", className)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: MdComponentProps) => (
    <h3 className={cn("text-lg font-bold mt-3 mb-1", className)} {...props}>
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: MdComponentProps) => (
    <p className={cn("mb-3 leading-7", className)} {...props}>
      {children}
    </p>
  ),
  a: ({ className, children, href, ...props }: MdComponentProps) => (
    <Badge className="text-xs mx-0.5">
      <a
        className={cn("text-blue-400 hover:text-blue-300 text-xs", className)}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        {...props}
      >
        {children}
      </a>
    </Badge>
  ),
  ul: ({ className, children, ...props }: MdComponentProps) => (
    <ul className={cn("list-disc pl-6 mb-3", className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: MdComponentProps) => (
    <ol className={cn("list-decimal pl-6 mb-3", className)} {...props}>
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: MdComponentProps) => (
    <li className={cn("mb-1", className)} {...props}>
      {children}
    </li>
  ),
  blockquote: ({ className, children, ...props }: MdComponentProps) => (
    <blockquote
      className={cn(
        "border-l-4 border-neutral-600 pl-4 italic my-3 text-sm",
        className
      )}
      {...props}
    >
      {children}
    </blockquote>
  ),
  code: ({ className, children, ...props }: MdComponentProps) => (
    <code
      className={cn(
        "bg-neutral-900 rounded px-1 py-0.5 font-mono text-xs",
        className
      )}
      {...props}
    >
      {children}
    </code>
  ),
  pre: ({ className, children, ...props }: MdComponentProps) => (
    <pre
      className={cn(
        "bg-neutral-900 p-3 rounded-lg overflow-x-auto font-mono text-xs my-3",
        className
      )}
      {...props}
    >
      {children}
    </pre>
  ),
  hr: ({ className, ...props }: MdComponentProps) => (
    <hr className={cn("border-neutral-600 my-4", className)} {...props} />
  ),
  table: ({ className, children, ...props }: MdComponentProps) => (
    <div className="my-3 overflow-x-auto">
      <table className={cn("border-collapse w-full", className)} {...props}>
        {children}
      </table>
    </div>
  ),
  th: ({ className, children, ...props }: MdComponentProps) => (
    <th
      className={cn(
        "border border-neutral-600 px-3 py-2 text-left font-bold",
        className
      )}
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ className, children, ...props }: MdComponentProps) => (
    <td
      className={cn("border border-neutral-600 px-3 py-2", className)}
      {...props}
    >
      {children}
    </td>
  ),
};

// Props for HumanMessageBubble
interface HumanMessageBubbleProps {
  message: Message;
  mdComponents: typeof mdComponents;
}

// Format user message for display (hide PDF base64 data)
function formatUserMessage(content: string): string {
  // Strip PDF base64 tag from anywhere in the message
  const stripped = content
    .replace(/\[UPLOAD_PDF:.+?\].+?\[\/UPLOAD_PDF\]/gs, "")
    .trim();
  const pdfMatch = content.match(/\[UPLOAD_PDF:(.+?)\]/);
  if (pdfMatch) {
    const prefix = stripped ? `${stripped}\n\n` : "";
    return `${prefix}📄 Uploaded: **${pdfMatch[1]}**`;
  }
  return content;
}

// HumanMessageBubble Component
const HumanMessageBubble: React.FC<HumanMessageBubbleProps> = ({
  message,
  mdComponents,
}) => {
  const displayContent =
    typeof message.content === "string"
      ? formatUserMessage(message.content)
      : JSON.stringify(message.content);
  return (
    <div
      className={`text-white rounded-3xl break-words min-h-7 bg-neutral-700 max-w-[100%] sm:max-w-[90%] px-4 pt-3 rounded-br-lg`}
    >
      <ReactMarkdown components={mdComponents}>
        {displayContent}
      </ReactMarkdown>
    </div>
  );
};

// Extract memory indicator section from message content
function extractMemorySection(content: string): { main: string; memory: string | null } {
  const marker = "\n\n---\n> ";
  const idx = content.indexOf(marker);
  if (idx === -1) return { main: content, memory: null };
  return {
    main: content.slice(0, idx),
    memory: content.slice(idx + marker.length).replace(/^> /gm, ""),
  };
}

interface MemoryItem {
  type: "archival" | "recall";
  source: string;
  preview: string;
  score: number;
  used: boolean;
}

function parseMemoryItems(text: string): { count: number; items: MemoryItem[] } {
  const items: MemoryItem[] = [];
  const lineRegex = /- \[(archival|recall):([^\]]*)\]\s*(.+?)\s*\(score:\s*([\d.]+)\)\s*(✓)?/g;
  let m;
  while ((m = lineRegex.exec(text)) !== null) {
    items.push({
      type: m[1] as "archival" | "recall",
      source: m[2],
      preview: m[3].trim(),
      score: parseFloat(m[4]),
      used: m[5] === "✓",
    });
  }
  return { count: items.length, items };
}

const MemoryBadge: React.FC<{ memoryText: string }> = ({ memoryText }) => {
  const { count, items: rawItems } = parseMemoryItems(memoryText);
  const items = [...rawItems].sort((a, b) => b.score - a.score);
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-3 rounded-lg border border-emerald-700/40 bg-emerald-950/20 overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-1.5 bg-emerald-950/40 hover:bg-emerald-950/60 transition-colors cursor-pointer"
      >
        <BookMarked className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
        <span className="text-emerald-400 text-xs font-medium">
          {count > 0
            ? `检索到 ${count} 条记忆`
            : "Memory Retrieved"}
        </span>
        <span className="ml-auto text-emerald-600">
          {expanded ? (
            <ChevronUp className="h-3.5 w-3.5" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5" />
          )}
        </span>
      </button>
      {expanded && items.length > 0 && (
        <div className="border-t border-emerald-800/30 overflow-hidden">
          {items.map((item, i) => (
            <div
              key={i}
              className={cn(
                "px-3 py-1.5 flex items-center gap-2 text-xs min-w-0",
                i > 0 && "border-t border-emerald-900/20",
              )}
            >
              <span className="shrink-0 w-5 text-right text-neutral-500 tabular-nums">
                {i + 1}
              </span>
              <span
                className={cn(
                  "shrink-0 text-[10px] px-1 py-0.5 rounded",
                  item.type === "archival"
                    ? "bg-blue-500/15 text-blue-400"
                    : "bg-amber-500/15 text-amber-400",
                )}
              >
                {item.type}
              </span>
              <span className="flex-1 min-w-0 truncate text-neutral-300">
                {item.preview}
              </span>
              <span className="shrink-0 text-neutral-500 tabular-nums">
                {(item.score * 100).toFixed(0)}%
              </span>
              {item.used && (
                <span className="shrink-0 text-emerald-400 text-sm font-bold">✓</span>
              )}
            </div>
          ))}
        </div>
      )}
      {expanded && items.length === 0 && (
        <div className="px-3 py-1.5 text-xs text-emerald-300/60 border-t border-emerald-800/30">
          {memoryText}
        </div>
      )}
    </div>
  );
};

// Props for AiMessageBubble
interface AiMessageBubbleProps {
  message: Message;
  historicalActivity: ProcessedEvent[] | undefined;
  liveActivity: ProcessedEvent[] | undefined;
  isLastMessage: boolean;
  isOverallLoading: boolean;
  mdComponents: typeof mdComponents;
  handleCopy: (text: string, messageId: string) => void;
  copiedMessageId: string | null;
}

// AiMessageBubble Component
const AiMessageBubble: React.FC<AiMessageBubbleProps> = ({
  message,
  historicalActivity,
  liveActivity,
  isLastMessage,
  isOverallLoading,
  mdComponents,
  handleCopy,
  copiedMessageId,
}) => {
  // Determine which activity events to show and if it's for a live loading message
  const activityForThisBubble =
    isLastMessage && isOverallLoading ? liveActivity : historicalActivity;
  const isLiveActivityForThisBubble = isLastMessage && isOverallLoading;

  const rawContent = typeof message.content === "string"
    ? message.content
    : JSON.stringify(message.content);
  const { main: mainContent, memory: memoryText } = extractMemorySection(rawContent);
  const memoryItems = memoryText ? parseMemoryItems(memoryText) : null;
  const hasMemoryResults = memoryItems && memoryItems.items.length > 0;

  return (
    <div className={`relative break-words flex flex-col min-w-0`}>
      {activityForThisBubble && activityForThisBubble.length > 0 && (
        <div className="mb-3 border-b border-neutral-700 pb-3 text-xs">
          <ActivityTimeline
            processedEvents={activityForThisBubble}
            isLoading={isLiveActivityForThisBubble}
          />
        </div>
      )}
      <ReactMarkdown components={mdComponents}>
        {mainContent}
      </ReactMarkdown>
      {hasMemoryResults && <MemoryBadge memoryText={memoryText!} />}
      <Button
        variant="default"
        className={`cursor-pointer bg-neutral-700 border-neutral-600 text-neutral-300 self-end ${
          message.content.length > 0 ? "visible" : "hidden"
        }`}
        onClick={() =>
          handleCopy(
            typeof message.content === "string"
              ? message.content
              : JSON.stringify(message.content),
            message.id!
          )
        }
      >
        {copiedMessageId === message.id ? "Copied" : "Copy"}
        {copiedMessageId === message.id ? <CopyCheck /> : <Copy />}
      </Button>
    </div>
  );
};

interface ChatMessagesViewProps {
  messages: Message[];
  isLoading: boolean;
  scrollAreaRef: React.RefObject<HTMLDivElement | null>;
  onSubmit: (inputValue: string, mode: string) => void;
  onCancel: () => void;
  liveActivityEvents: ProcessedEvent[];
  historicalActivities: Record<string, ProcessedEvent[]>;
}

export function ChatMessagesView({
  messages,
  isLoading,
  scrollAreaRef,
  onSubmit,
  onCancel,
  liveActivityEvents,
  historicalActivities,
}: ChatMessagesViewProps) {
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  const handleCopy = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };
  return (
    <div className="flex flex-col h-full">
      <div className="absolute top-4 right-4 z-10 flex items-center gap-3">
        <Link
          to="/analytics"
          className="flex items-center gap-1.5 text-sm text-neutral-400 hover:text-neutral-100 transition-colors"
        >
          <BarChart3 className="w-4 h-4" />
          Analytics
        </Link>
        <Link
          to="/library"
          className="flex items-center gap-1.5 text-sm text-neutral-400 hover:text-neutral-100 transition-colors"
        >
          <BookOpen className="w-4 h-4" />
          Library
        </Link>
      </div>
      <ScrollArea className="flex-1 overflow-y-auto overflow-x-hidden" ref={scrollAreaRef}>
        <div className="p-4 md:p-6 space-y-2 max-w-4xl mx-auto pt-16">
          {messages.filter((message) => {
            // Skip AI messages with no content (intermediate tool-calling steps)
            if (message.type === "ai") {
              const c = message.content;
              if (!c || (typeof c === "string" && c.trim().length === 0)) return false;
            }
            return true;
          }).map((message, index, arr) => {
            const isLast = index === arr.length - 1;
            return (
              <div key={message.id || `msg-${index}`} className="space-y-3">
                <div
                  className={`flex items-start gap-3 ${
                    message.type === "human" ? "justify-end" : ""
                  }`}
                >
                  {message.type === "human" ? (
                    <HumanMessageBubble
                      message={message}
                      mdComponents={mdComponents}
                    />
                  ) : (
                    <AiMessageBubble
                      message={message}
                      historicalActivity={historicalActivities[message.id!]}
                      liveActivity={liveActivityEvents} // Pass global live events
                      isLastMessage={isLast}
                      isOverallLoading={isLoading} // Pass global loading state
                      mdComponents={mdComponents}
                      handleCopy={handleCopy}
                      copiedMessageId={copiedMessageId}
                    />
                  )}
                </div>
              </div>
            );
          })}
          {isLoading && (
            <div className="flex items-start gap-3 mt-3">
              <div className="relative group max-w-[85%] md:max-w-[80%] rounded-xl p-3 shadow-sm break-words bg-neutral-800 text-neutral-100 rounded-bl-none w-full min-h-[56px]">
                <div className="text-xs">
                  <ActivityTimeline
                    processedEvents={liveActivityEvents}
                    isLoading={true}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
      <InputForm
        onSubmit={onSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={messages.length > 0}
      />
    </div>
  );
}
