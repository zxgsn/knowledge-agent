import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ArrowLeft, Search, BookOpen, MessageSquare, RefreshCw } from "lucide-react";

interface ArchivalEntry {
  id: string;
  namespace: string;
  content: string;
  metadata: Record<string, any>;
  created_at: string | null;
}

interface RecallEntry {
  id: string;
  thread_id: string;
  role: string;
  content: string;
  metadata: Record<string, any>;
  created_at: string | null;
}

interface NamespaceStats {
  namespace: string;
  count: number;
  oldest: string | null;
  newest: string | null;
}

const NAMESPACE_COLORS: Record<string, string> = {
  ingested: "bg-sky-600",
  research: "bg-emerald-600",
  manual: "bg-amber-600",
  conversation_facts: "bg-purple-600",
  default: "bg-neutral-600",
};

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return "";
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
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "...";
}

export function DocumentLibrary() {
  const [activeTab, setActiveTab] = useState("all");
  const [search, setSearch] = useState("");
  const [entries, setEntries] = useState<ArchivalEntry[]>([]);
  const [recallEntries, setRecallEntries] = useState<RecallEntry[]>([]);
  const [stats, setStats] = useState<NamespaceStats[]>([]);
  const [recallStats, setRecallStats] = useState({ message_count: 0, thread_count: 0 });
  const [selectedEntry, setSelectedEntry] = useState<ArchivalEntry | RecallEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);

  const fetchStats = useCallback(async () => {
    try {
      const [archivalRes, recallRes] = await Promise.all([
        fetch("/api/archival/stats"),
        fetch("/api/recall/stats"),
      ]);
      if (archivalRes.ok) setStats(await archivalRes.json());
      if (recallRes.ok) setRecallStats(await recallRes.json());
    } catch { /* ignore */ }
  }, []);

  const fetchEntries = useCallback(async () => {
    setLoading(true);
    try {
      if (activeTab === "recall") {
        const params = new URLSearchParams({ limit: "100", offset: "0" });
        if (search) params.set("search", search);
        const res = await fetch(`/api/recall/entries?${params}`);
        if (res.ok) {
          const data = await res.json();
          setRecallEntries(data.entries);
          setTotal(data.total);
        }
        setEntries([]);
      } else {
        const params = new URLSearchParams({ limit: "100", offset: "0" });
        if (activeTab !== "all") params.set("namespace", activeTab);
        if (search) params.set("search", search);
        const res = await fetch(`/api/archival/entries?${params}`);
        if (res.ok) {
          const data = await res.json();
          setEntries(data.entries);
          setTotal(data.total);
        }
        setRecallEntries([]);
      }
    } catch { /* ignore */ }
    setLoading(false);
  }, [activeTab, search]);

  useEffect(() => { fetchStats(); }, [fetchStats]);
  useEffect(() => { fetchEntries(); }, [fetchEntries]);

  const handleRefresh = () => {
    fetchStats();
    fetchEntries();
  };

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100">
      <div className="flex flex-col w-full max-w-6xl mx-auto p-6 gap-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/">
              <Button variant="ghost" size="icon" className="text-neutral-400 hover:text-neutral-100">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <BookOpen className="w-6 h-6 text-neutral-400" />
            <h1 className="text-2xl font-semibold">Document Library</h1>
          </div>
          <Button variant="ghost" size="icon" onClick={handleRefresh} className="text-neutral-400 hover:text-neutral-100">
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>

        {/* Stats bar */}
        <div className="flex gap-3 flex-wrap">
          {stats.map((s) => (
            <Badge key={s.namespace} variant="outline" className="border-neutral-600 text-neutral-300">
              {s.namespace}: {s.count}
            </Badge>
          ))}
          <Badge variant="outline" className="border-neutral-600 text-neutral-300">
            recall: {recallStats.message_count} ({recallStats.thread_count} threads)
          </Badge>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-500" />
          <Input
            placeholder="Search content..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 bg-neutral-700 border-neutral-600 text-neutral-100 placeholder:text-neutral-500"
          />
        </div>

        {/* Tabs + Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
          <TabsList className="bg-neutral-700">
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="ingested">Ingested</TabsTrigger>
            <TabsTrigger value="research">Research</TabsTrigger>
            <TabsTrigger value="manual">Manual</TabsTrigger>
            <TabsTrigger value="recall">
              <MessageSquare className="w-3.5 h-3.5 mr-1" />
              Recall
            </TabsTrigger>
          </TabsList>

          <div className="flex-1 min-h-0 mt-3">
            <ScrollArea className="h-full">
              {loading ? (
                <div className="text-center text-neutral-500 py-12">Loading...</div>
              ) : activeTab === "recall" ? (
                recallEntries.length === 0 ? (
                  <div className="text-center text-neutral-500 py-12">No recall entries found.</div>
                ) : (
                  <div className="grid gap-2">
                    {recallEntries.map((entry) => (
                      <Card
                        key={entry.id}
                        className={`bg-neutral-700 border-neutral-600 cursor-pointer hover:border-neutral-500 transition-colors ${
                          selectedEntry?.id === entry.id ? "border-neutral-400" : ""
                        }`}
                        onClick={() => setSelectedEntry(selectedEntry?.id === entry.id ? null : entry)}
                      >
                        <CardContent className="p-3">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge className={`text-xs ${
                              entry.role === "user" ? "bg-sky-600" : "bg-emerald-600"
                            }`}>
                              {entry.role}
                            </Badge>
                            <span className="text-xs text-neutral-500">
                              thread: {entry.thread_id?.slice(0, 8)}
                            </span>
                            <span className="text-xs text-neutral-500 ml-auto">
                              {timeAgo(entry.created_at)}
                            </span>
                          </div>
                          <p className="text-sm text-neutral-200">
                            {selectedEntry?.id === entry.id
                              ? entry.content
                              : truncate(entry.content, 200)}
                          </p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )
              ) : entries.length === 0 ? (
                <div className="text-center text-neutral-500 py-12">No entries found.</div>
              ) : (
                <div className="grid gap-2">
                  {entries.map((entry) => (
                    <Card
                      key={entry.id}
                      className={`bg-neutral-700 border-neutral-600 cursor-pointer hover:border-neutral-500 transition-colors ${
                        selectedEntry?.id === entry.id ? "border-neutral-400" : ""
                      }`}
                      onClick={() => setSelectedEntry(selectedEntry?.id === entry.id ? null : entry)}
                    >
                      <CardContent className="p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge className={`text-xs ${NAMESPACE_COLORS[entry.namespace] || NAMESPACE_COLORS.default}`}>
                            {entry.namespace}
                          </Badge>
                          {entry.metadata?.source && (
                            <span className="text-xs text-neutral-500">
                              {entry.metadata.source}
                            </span>
                          )}
                          {entry.metadata?.document && (
                            <span className="text-xs text-neutral-500">
                              {entry.metadata.document}
                            </span>
                          )}
                          <span className="text-xs text-neutral-500 ml-auto">
                            {timeAgo(entry.created_at)}
                          </span>
                        </div>
                        <p className="text-sm text-neutral-200">
                          {selectedEntry?.id === entry.id
                            ? entry.content
                            : truncate(entry.content, 200)}
                        </p>
                        {selectedEntry?.id === entry.id && Object.keys(entry.metadata).length > 0 && (
                          <div className="mt-2 pt-2 border-t border-neutral-600">
                            <p className="text-xs text-neutral-500 mb-1">Metadata:</p>
                            <pre className="text-xs text-neutral-400 whitespace-pre-wrap break-all">
                              {JSON.stringify(entry.metadata, null, 2)}
                            </pre>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>

          <div className="text-xs text-neutral-500 mt-1">
            {total} entries
          </div>
        </Tabs>
      </div>
    </div>
  );
}
