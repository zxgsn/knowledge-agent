import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Loader2,
  Activity,
  Info,
  Search,
  TextSearch,
  Brain,
  Pen,
  Database,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useEffect, useState } from "react";

export interface ProcessedEvent {
  title: string;
  data: any;
}

interface ActivityTimelineProps {
  processedEvents: ProcessedEvent[];
  isLoading: boolean;
}

export function ActivityTimeline({
  processedEvents,
  isLoading,
}: ActivityTimelineProps) {
  const [isTimelineCollapsed, setIsTimelineCollapsed] =
    useState<boolean>(false);
  const getEventIcon = (title: string, index: number) => {
    if (index === 0 && isLoading && processedEvents.length === 0) {
      return <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />;
    }
    if (title.toLowerCase().includes("generating")) {
      return <TextSearch className="h-4 w-4 text-neutral-400" />;
    } else if (title.toLowerCase().includes("thinking")) {
      return <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />;
    } else if (title.toLowerCase().includes("reflection")) {
      return <Brain className="h-4 w-4 text-neutral-400" />;
    } else if (title.toLowerCase().includes("research")) {
      return <Search className="h-4 w-4 text-neutral-400" />;
    } else if (title.toLowerCase().includes("finalizing") || title.toLowerCase().includes("response")) {
      return <Pen className="h-4 w-4 text-neutral-400" />;
    } else if (title.toLowerCase().includes("memory sufficient")) {
      return <Database className="h-4 w-4 text-emerald-400" />;
    } else if (title.toLowerCase().includes("memory insufficient")) {
      return <Database className="h-4 w-4 text-yellow-400" />;
    } else if (title.toLowerCase().includes("memory pipeline") || title.toLowerCase().includes("memory consolidated") || title.toLowerCase().includes("memory cleanup")) {
      return <Database className="h-4 w-4 text-cyan-400" />;
    } else if (title.toLowerCase().includes("saved to archival") || title.toLowerCase().includes("document ingested")) {
      return <Database className="h-4 w-4 text-emerald-400" />;
    } else if (title.toLowerCase().includes("saving") || title.toLowerCase().includes("memory")) {
      return <Database className="h-4 w-4 text-emerald-400" />;
    } else if (title.toLowerCase().includes("intent")) {
      return <Brain className="h-4 w-4 text-neutral-400" />;
    }
    return <Activity className="h-4 w-4 text-neutral-400" />;
  };

  useEffect(() => {
    if (!isLoading && processedEvents.length !== 0) {
      setIsTimelineCollapsed(true);
    }
  }, [isLoading, processedEvents]);

  return (
    <Card className="border-none rounded-lg bg-neutral-700 max-h-96 overflow-hidden">
      <CardHeader>
        <CardDescription className="flex items-center justify-between">
          <div
            className="flex items-center justify-start text-sm w-full cursor-pointer gap-2 text-neutral-100"
            onClick={() => setIsTimelineCollapsed(!isTimelineCollapsed)}
          >
            {processedEvents.some(e => e.title.toLowerCase().includes("memory"))
              ? "Memory & Research"
              : "Research"}
            {isTimelineCollapsed ? (
              <ChevronDown className="h-4 w-4 mr-2" />
            ) : (
              <ChevronUp className="h-4 w-4 mr-2" />
            )}
          </div>
        </CardDescription>
      </CardHeader>
      {!isTimelineCollapsed && (
        <ScrollArea className="max-h-96 overflow-y-auto overflow-x-hidden">
          <CardContent>
            {isLoading && processedEvents.length === 0 && (
              <div className="relative pl-8 pb-4">
                <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-800" />
                <div className="absolute left-0.5 top-2 h-5 w-5 rounded-full bg-neutral-800 flex items-center justify-center ring-4 ring-neutral-900">
                  <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                </div>
                <div>
                  <p className="text-sm text-neutral-300 font-medium">
                    Searching...
                  </p>
                </div>
              </div>
            )}
            {processedEvents.length > 0 ? (
              <div className="space-y-0">
                {processedEvents.map((eventItem, index) => (
                  <div key={index} className="relative pl-8 pb-4">
                    {index < processedEvents.length - 1 ||
                    (isLoading && index === processedEvents.length - 1) ? (
                      <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-600" />
                    ) : null}
                    <div className="absolute left-0.5 top-2 h-6 w-6 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700">
                      {getEventIcon(eventItem.title, index)}
                    </div>
                    <div>
                      <p className="text-sm text-neutral-200 font-medium mb-0.5">
                        {eventItem.title}
                      </p>
                      {Array.isArray(eventItem.data) &&
                      eventItem.data.length > 0 &&
                      typeof eventItem.data[0] === "object" &&
                      "type" in eventItem.data[0] ? (
                        <div className="space-y-1.5 mt-1 overflow-hidden">
                          {[...eventItem.data]
                            .sort((a: any, b: any) => (b.score || 0) - (a.score || 0))
                            .map(
                            (item: any, i: number) => (
                              <div
                                key={i}
                                className="rounded-md bg-neutral-800/70 px-2.5 py-1.5 text-xs overflow-hidden"
                              >
                                <p className="text-neutral-200 leading-relaxed mb-1 break-words line-clamp-3">
                                  {item.content || ""}
                                </p>
                                <div className="flex flex-wrap gap-1.5">
                                  {item.type && (
                                    <span
                                      className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium ${
                                        item.type === "archival"
                                          ? "bg-emerald-900/60 text-emerald-300"
                                          : "bg-blue-900/60 text-blue-300"
                                      }`}
                                    >
                                      {item.type}
                                    </span>
                                  )}
                                  {item.sourceType && (
                                    <span className="inline-flex items-center rounded bg-neutral-700 px-1.5 py-0.5 text-[10px] text-neutral-300">
                                      {item.sourceType}
                                    </span>
                                  )}
                                  {item.document && (
                                    <span className="inline-flex items-center rounded bg-neutral-700 px-1.5 py-0.5 text-[10px] text-neutral-300">
                                      📄 {item.document}
                                    </span>
                                  )}
                                  {item.namespace && (
                                    <span className="inline-flex items-center rounded bg-neutral-700 px-1.5 py-0.5 text-[10px] text-neutral-300">
                                      {item.namespace}
                                    </span>
                                  )}
                                  {item.timestamp && (
                                    <span className="inline-flex items-center rounded bg-neutral-700 px-1.5 py-0.5 text-[10px] text-neutral-400">
                                      {new Date(item.timestamp).toLocaleDateString()}
                                    </span>
                                  )}
                                  {item.role && (
                                    <span className="inline-flex items-center rounded bg-neutral-700 px-1.5 py-0.5 text-[10px] text-neutral-300">
                                      {item.role}
                                    </span>
                                  )}
                                  {item.score > 0 && (
                                    <span className="inline-flex items-center rounded bg-neutral-700 px-1.5 py-0.5 text-[10px] text-neutral-400">
                                      {item.score.toFixed(3)}
                                    </span>
                                  )}
                                  {item.used && (
                                    <span className="inline-flex items-center text-emerald-400 text-sm font-bold">✓</span>
                                  )}
                                </div>
                              </div>
                            )
                          )}
                        </div>
                      ) : (
                        <p className="text-xs text-neutral-300 leading-relaxed whitespace-pre-line">
                          {typeof eventItem.data === "string"
                            ? eventItem.data
                            : Array.isArray(eventItem.data)
                            ? (eventItem.data as string[]).join("\n")
                            : JSON.stringify(eventItem.data)}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
                {isLoading && processedEvents.length > 0 && (
                  <div className="relative pl-8 pb-4">
                    <div className="absolute left-0.5 top-2 h-5 w-5 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700">
                      <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                    </div>
                    <div>
                      <p className="text-sm text-neutral-300 font-medium">
                        Searching...
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : !isLoading ? ( // Only show "No activity" if not loading and no events
              <div className="flex flex-col items-center justify-center h-full text-neutral-500 pt-10">
                <Info className="h-6 w-6 mb-3" />
                <p className="text-sm">No activity to display.</p>
                <p className="text-xs text-neutral-600 mt-1">
                  Timeline will update during processing.
                </p>
              </div>
            ) : null}
          </CardContent>
        </ScrollArea>
      )}
    </Card>
  );
}
