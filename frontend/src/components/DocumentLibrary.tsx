import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ArrowLeft, Search, BookOpen, MessageSquare, RefreshCw, FileText, ChevronDown, ChevronRight, Trash2, Plus, Pencil } from "lucide-react";

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

interface DocumentSummary {
  id: string;
  title: string;
  source: string | null;
  source_type: string | null;
  chunk_count: number;
  created_at: string | null;
}

interface DocumentDetail extends DocumentSummary {
  content_full: string;
  metadata: Record<string, any>;
  chunks: ArchivalEntry[];
}

const NAMESPACE_COLORS: Record<string, string> = {
  ingested: "bg-sky-600",
  research: "bg-emerald-600",
  manual: "bg-amber-600",
  conversation_facts: "bg-purple-600",
  default: "bg-neutral-600",
};

const API_BASE = import.meta.env.DEV
  ? "http://localhost:8000"
  : "";

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
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<DocumentDetail | null>(null);
  const [stats, setStats] = useState<NamespaceStats[]>([]);
  const [recallStats, setRecallStats] = useState({ message_count: 0, thread_count: 0 });
  const [selectedEntry, setSelectedEntry] = useState<ArchivalEntry | RecallEntry | null>(null);
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingEntryId, setEditingEntryId] = useState<string | null>(null);
  const [formContent, setFormContent] = useState("");
  const [formMetadata, setFormMetadata] = useState("{}");

  const fetchStats = useCallback(async () => {
    try {
      const [archivalRes, recallRes] = await Promise.all([
        fetch(`${API_BASE}/api/archival/stats`),
        fetch(`${API_BASE}/api/recall/stats`),
      ]);
      if (archivalRes.ok) setStats(await archivalRes.json());
      if (recallRes.ok) setRecallStats(await recallRes.json());
    } catch { /* ignore */ }
  }, []);

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: "100", offset: "0" });
      if (search) params.set("search", search);
      const res = await fetch(`${API_BASE}/api/documents?${params}`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents);
        setTotal(data.total);
      }
    } catch { /* ignore */ }
    setLoading(false);
  }, [search]);

  const fetchDocDetail = useCallback(async (docId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/documents/${docId}`);
      if (res.ok) {
        setSelectedDoc(await res.json());
      }
    } catch { /* ignore */ }
  }, []);

  const fetchEntries = useCallback(async () => {
    setLoading(true);
    try {
      if (activeTab === "recall") {
        const params = new URLSearchParams({ limit: "100", offset: "0" });
        if (search) params.set("search", search);
        const res = await fetch(`${API_BASE}/api/recall/entries?${params}`);
        if (res.ok) {
          const data = await res.json();
          setRecallEntries(data.entries);
          setTotal(data.total);
        }
        setEntries([]);
        setDocuments([]);
      } else if (activeTab === "all") {
        // Fetch both archival (filtered) and recall in parallel
        const archParams = new URLSearchParams({ limit: "50", offset: "0", exclude_namespaces: EXCLUDE_NS_PARAM });
        const recParams = new URLSearchParams({ limit: "50", offset: "0" });
        if (search) { archParams.set("search", search); recParams.set("search", search); }
        const [archRes, recRes] = await Promise.all([
          fetch(`${API_BASE}/api/archival/entries?${archParams}`),
          fetch(`${API_BASE}/api/recall/entries?${recParams}`),
        ]);
        let archTotal = 0;
        if (archRes.ok) {
          const data = await archRes.json();
          setEntries(data.entries);
          archTotal = data.total;
        }
        if (recRes.ok) {
          const data = await recRes.json();
          setRecallEntries(data.entries);
          setTotal(archTotal + data.total);
        } else {
          setTotal(archTotal);
        }
        setDocuments([]);
      } else {
        const params = new URLSearchParams({ limit: "100", offset: "0" });
        params.set("namespace", activeTab);
        if (search) params.set("search", search);
        const res = await fetch(`${API_BASE}/api/archival/entries?${params}`);
        if (res.ok) {
          const data = await res.json();
          setEntries(data.entries);
          setTotal(data.total);
        }
        setRecallEntries([]);
        setDocuments([]);
      }
    } catch { /* ignore */ }
    setLoading(false);
  }, [activeTab, search]);

  const HIDDEN_NAMESPACES = [
    "locomo_baseline", "locomo_context", "locomo_extracted",
    "locomo_bench_baseline", "locomo_bench_context",
    "ingested", "conversation_facts", "session_context", "test",
  ];
  const EXCLUDE_NS_PARAM = HIDDEN_NAMESPACES.join(",");

  const visibleStats = stats.filter((s) => !HIDDEN_NAMESPACES.includes(s.namespace));

  const visibleEntries = entries.filter((e) => !HIDDEN_NAMESPACES.includes(e.namespace));

  useEffect(() => { fetchStats(); }, [fetchStats]);
  useEffect(() => {
    if (activeTab === "ingested") {
      fetchDocuments();
    } else {
      fetchEntries();
    }
  }, [activeTab, search, fetchDocuments, fetchEntries]);

  const handleRefresh = () => {
    fetchStats();
    if (activeTab === "ingested") {
      fetchDocuments();
    } else {
      fetchEntries();
    }
  };

  const handleDocClick = (doc: DocumentSummary) => {
    if (selectedDoc?.id === doc.id) {
      setSelectedDoc(null);
      setSelectedChunkId(null);
    } else {
      setSelectedChunkId(null);
      fetchDocDetail(doc.id);
    }
  };

  const handleDeleteDocument = async (e: React.MouseEvent, docId: string, title: string) => {
    e.stopPropagation();
    if (!window.confirm(`Delete "${title}" and all its chunks?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/documents/${docId}`, { method: "DELETE" });
      if (res.ok || res.status === 404) {
        setSelectedDoc(null);
        fetchDocuments();
        fetchStats();
      } else {
        alert(`Delete failed (${res.status}): ${await res.text()}`);
      }
    } catch (err) {
      alert(`Delete failed: ${err}`);
    }
  };

  const handleDeleteArchival = async (e: React.MouseEvent, entryId: string) => {
    e.stopPropagation();
    if (!window.confirm("Delete this entry?")) return;
    try {
      const res = await fetch(`${API_BASE}/api/archival/entries/${entryId}`, { method: "DELETE" });
      if (res.ok || res.status === 404) {
        setEntries((prev) => prev.filter((e) => e.id !== entryId));
        setSelectedEntry(null);
        fetchStats();
      } else {
        alert(`Delete failed (${res.status}): ${await res.text()}`);
      }
    } catch (err) {
      alert(`Delete failed: ${err}`);
    }
  };

  const handleDeleteRecall = async (e: React.MouseEvent, entryId: string) => {
    e.stopPropagation();
    if (!window.confirm("Delete this recall entry?")) return;
    try {
      const res = await fetch(`${API_BASE}/api/recall/entries/${entryId}`, { method: "DELETE" });
      if (res.ok || res.status === 404) {
        setRecallEntries((prev) => prev.filter((e) => e.id !== entryId));
        setSelectedEntry(null);
        fetchStats();
      } else {
        alert(`Delete failed (${res.status}): ${await res.text()}`);
      }
    } catch (err) {
      alert(`Delete failed: ${err}`);
    }
  };

  const handleCreateEntry = async () => {
    if (!formContent.trim()) return;
    let metadata = {};
    try {
      metadata = JSON.parse(formMetadata || "{}");
    } catch {
      alert("Metadata must be valid JSON");
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/api/archival/entries`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: formContent, namespace: "manual", metadata }),
      });
      if (res.ok) {
        const entry = await res.json();
        setEntries((prev) => [entry, ...prev]);
        setShowCreateForm(false);
        setFormContent("");
        setFormMetadata("{}");
        fetchStats();
      } else {
        alert(`Create failed (${res.status}): ${await res.text()}`);
      }
    } catch (err) {
      alert(`Create failed: ${err}`);
    }
  };

  const handleUpdateEntry = async (entryId: string) => {
    if (!formContent.trim()) return;
    let metadata: Record<string, unknown> | undefined;
    if (formMetadata.trim()) {
      try {
        metadata = JSON.parse(formMetadata);
      } catch {
        alert("Metadata must be valid JSON");
        return;
      }
    }
    try {
      const res = await fetch(`${API_BASE}/api/archival/entries/${entryId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: formContent, metadata }),
      });
      if (res.ok) {
        const updated = await res.json();
        setEntries((prev) => prev.map((e) => (e.id === entryId ? updated : e)));
        setEditingEntryId(null);
        setFormContent("");
        setFormMetadata("{}");
      } else {
        alert(`Update failed (${res.status}): ${await res.text()}`);
      }
    } catch (err) {
      alert(`Update failed: ${err}`);
    }
  };

  const startEdit = (entry: ArchivalEntry) => {
    setEditingEntryId(entry.id);
    setFormContent(entry.content);
    setFormMetadata(JSON.stringify(entry.metadata, null, 2));
  };

  return (
    <div className="flex justify-center w-full h-screen bg-neutral-800 text-neutral-100 overflow-auto">
      <div className="flex flex-col w-[1024px] max-w-[calc(100vw-3rem)] p-6 gap-4">
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
          {visibleStats.map((s) => (
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
            <TabsTrigger value="ingested">Documents</TabsTrigger>
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
              ) : activeTab === "ingested" ? (
                /* Document-level view for ingested tab */
                documents.length === 0 ? (
                  <div className="text-center text-neutral-500 py-12">No documents found.</div>
                ) : (
                  <div className="grid gap-2">
                    {documents.map((doc) => (
                      <Card
                        key={doc.id}
                        className={`bg-neutral-700 border-neutral-600 cursor-pointer hover:border-neutral-500 transition-colors ${
                          selectedDoc?.id === doc.id ? "border-neutral-400" : ""
                        }`}
                        onClick={() => handleDocClick(doc)}
                      >
                        <CardContent className="p-3">
                          <div className="flex items-center gap-2 mb-1">
                            <FileText className="w-4 h-4 text-sky-400" />
                            <span className="text-sm font-medium text-neutral-100">
                              {doc.title}
                            </span>
                            <Badge variant="outline" className="text-xs border-neutral-600 text-neutral-400">
                              {doc.chunk_count} chunks
                            </Badge>
                            {doc.source_type && (
                              <Badge variant="outline" className="text-xs border-neutral-600 text-neutral-400">
                                {doc.source_type}
                              </Badge>
                            )}
                            <span className="text-xs text-neutral-500 ml-auto">
                              {timeAgo(doc.created_at)}
                            </span>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-neutral-500 hover:text-red-400"
                              onClick={(e) => handleDeleteDocument(e, doc.id, doc.title)}
                              title="Delete document"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </Button>
                            {selectedDoc?.id === doc.id
                              ? <ChevronDown className="w-4 h-4 text-neutral-400" />
                              : <ChevronRight className="w-4 h-4 text-neutral-400" />}
                          </div>
                          {doc.source && (
                            <p className="text-xs text-neutral-500 mb-1 truncate">{doc.source}</p>
                          )}

                          {/* Expanded: full content + chunks */}
                          {selectedDoc?.id === doc.id && selectedDoc && (
                            <div className="mt-3 pt-3 border-t border-neutral-600">
                              <div className="mb-3">
                                <p className="text-xs text-neutral-500 mb-1">Full Document:</p>
                                <div className="text-sm text-neutral-200 max-h-64 overflow-y-auto whitespace-pre-wrap bg-neutral-800 rounded p-2">
                                  {selectedDoc.content_full}
                                </div>
                              </div>
                              {selectedDoc.chunks.length > 0 && (
                                <div>
                                  <p className="text-xs text-neutral-500 mb-1">
                                    Chunks ({selectedDoc.chunks.length}):
                                  </p>
                                  <div className="grid gap-1">
                                    {selectedDoc.chunks.map((chunk, i) => (
                                      <div
                                        key={chunk.id}
                                        className="text-xs text-neutral-300 bg-neutral-800 rounded p-2 cursor-pointer hover:bg-neutral-750 transition-colors"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setSelectedChunkId(selectedChunkId === chunk.id ? null : chunk.id);
                                        }}
                                      >
                                        <div className="flex items-start gap-2">
                                          <span className="text-neutral-500 shrink-0">#{i + 1}</span>
                                          <span className="flex-1">
                                            {selectedChunkId === chunk.id
                                              ? chunk.content
                                              : truncate(chunk.content, 150)}
                                          </span>
                                          <span className="text-neutral-600 shrink-0">
                                            {selectedChunkId === chunk.id ? "▲" : "▼"}
                                          </span>
                                        </div>
                                        {selectedChunkId === chunk.id && Object.keys(chunk.metadata).length > 0 && (
                                          <div className="mt-2 pt-2 border-t border-neutral-700">
                                            <pre className="text-xs text-neutral-400 whitespace-pre-wrap break-all">
                                              {JSON.stringify(chunk.metadata, null, 2)}
                                            </pre>
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )
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
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-neutral-500 hover:text-red-400"
                              onClick={(e) => handleDeleteRecall(e, entry.id)}
                              title="Delete entry"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </Button>
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
              ) : visibleEntries.length === 0 && recallEntries.length === 0 && !showCreateForm ? (
                <div className="text-center text-neutral-500 py-12">No entries found.</div>
              ) : (
                <div className="grid gap-2">
                  {/* Add Entry button (Manual tab only) */}
                  {activeTab === "manual" && (
                    <>
                      {!showCreateForm ? (
                        <Button
                          variant="outline"
                          className="border-dashed border-neutral-600 text-neutral-400 hover:text-neutral-200 hover:border-neutral-500 justify-start"
                          onClick={() => { setShowCreateForm(true); setFormContent(""); setFormMetadata("{}"); }}
                        >
                          <Plus className="w-4 h-4 mr-2" /> Add Entry
                        </Button>
                      ) : (
                        <Card className="bg-neutral-700 border-amber-600/50">
                          <CardContent className="p-3 space-y-2">
                            <p className="text-xs text-neutral-400 font-medium">New Entry</p>
                            <textarea
                              value={formContent}
                              onChange={(e) => setFormContent(e.target.value)}
                              placeholder="Content..."
                              className="w-full bg-neutral-800 text-neutral-200 text-sm rounded p-2 min-h-[80px] resize-y border border-neutral-600 focus:border-amber-600 outline-none"
                            />
                            <Input
                              value={formMetadata}
                              onChange={(e) => setFormMetadata(e.target.value)}
                              placeholder='Metadata (JSON, e.g. {"source": "manual"})'
                              className="bg-neutral-800 border-neutral-600 text-neutral-200 text-xs h-8"
                            />
                            <div className="flex gap-2 justify-end">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-neutral-400"
                                onClick={() => { setShowCreateForm(false); setFormContent(""); setFormMetadata("{}"); }}
                              >
                                Cancel
                              </Button>
                              <Button
                                size="sm"
                                className="bg-amber-600 hover:bg-amber-700 text-white"
                                onClick={handleCreateEntry}
                                disabled={!formContent.trim()}
                              >
                                Save
                              </Button>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </>
                  )}
                  {visibleEntries.map((entry) => (
                    <Card
                      key={entry.id}
                      className={`bg-neutral-700 border-neutral-600 cursor-pointer hover:border-neutral-500 transition-colors ${
                        selectedEntry?.id === entry.id ? "border-neutral-400" : ""
                      }`}
                      onClick={() => { if (editingEntryId !== entry.id) setSelectedEntry(selectedEntry?.id === entry.id ? null : entry); }}
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
                          {activeTab === "manual" && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-neutral-500 hover:text-amber-400"
                              onClick={(e) => { e.stopPropagation(); startEdit(entry); }}
                              title="Edit entry"
                            >
                              <Pencil className="w-3.5 h-3.5" />
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-neutral-500 hover:text-red-400"
                            onClick={(e) => handleDeleteArchival(e, entry.id)}
                            title="Delete entry"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                        {editingEntryId === entry.id ? (
                          <div className="space-y-2 mt-2">
                            <textarea
                              value={formContent}
                              onChange={(e) => setFormContent(e.target.value)}
                              className="w-full bg-neutral-800 text-neutral-200 text-sm rounded p-2 min-h-[80px] resize-y border border-neutral-600 focus:border-amber-600 outline-none"
                            />
                            <Input
                              value={formMetadata}
                              onChange={(e) => setFormMetadata(e.target.value)}
                              placeholder='Metadata (JSON)'
                              className="bg-neutral-800 border-neutral-600 text-neutral-200 text-xs h-8"
                            />
                            <div className="flex gap-2 justify-end">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-neutral-400"
                                onClick={() => { setEditingEntryId(null); setFormContent(""); setFormMetadata("{}"); }}
                              >
                                Cancel
                              </Button>
                              <Button
                                size="sm"
                                className="bg-amber-600 hover:bg-amber-700 text-white"
                                onClick={() => handleUpdateEntry(entry.id)}
                                disabled={!formContent.trim()}
                              >
                                Save
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <>
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
                          </>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                  {/* Recall entries in "All" tab */}
                  {activeTab === "all" && recallEntries.map((entry) => (
                    <Card
                      key={entry.id}
                      className={`bg-neutral-700 border-neutral-600 cursor-pointer hover:border-neutral-500 transition-colors ${
                        selectedEntry?.id === entry.id ? "border-neutral-400" : ""
                      }`}
                      onClick={() => setSelectedEntry(selectedEntry?.id === entry.id ? null : entry)}
                    >
                      <CardContent className="p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <MessageSquare className="w-3.5 h-3.5 text-neutral-400" />
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
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-neutral-500 hover:text-red-400"
                            onClick={(e) => handleDeleteRecall(e, entry.id)}
                            title="Delete entry"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
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
              )}
            </ScrollArea>
          </div>

          <div className="text-xs text-neutral-500 mt-1">
            {activeTab === "ingested" ? `${documents.length} documents` : `${total} entries`}
          </div>
        </Tabs>
      </div>
    </div>
  );
}
