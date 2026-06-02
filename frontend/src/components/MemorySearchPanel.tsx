import { useState, useCallback, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LIBRARY_API_BASE } from "@/lib/api";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ArrowLeft, Search, SlidersHorizontal, Loader2, ChevronDown, ChevronUp, SortAsc, SortDesc } from "lucide-react";

const API_BASE = LIBRARY_API_BASE;

interface SearchResult {
  id: string;
  content: string;
  namespace: string;
  score: number;
  cosine_score: number;
  importance_score: number;
  created_at: string | null;
  metadata: Record<string, any>;
}

interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
}

const NAMESPACE_COLORS: Record<string, string> = {
  ingested: "bg-sky-600",
  research: "bg-emerald-600",
  manual: "bg-amber-600",
  conversation_facts: "bg-purple-600",
};

function getImportanceColor(score: number): string {
  if (score >= 0.8) return "bg-emerald-500";
  if (score >= 0.6) return "bg-sky-500";
  if (score >= 0.4) return "bg-amber-500";
  return "bg-neutral-500";
}

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

type SortField = "relevance" | "date" | "importance";
type SortDir = "asc" | "desc";

export function MemorySearchPanel() {
  const [query, setQuery] = useState("");
  const [namespace, setNamespace] = useState("all");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>("relevance");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [offset, setOffset] = useState(0);
  const [showFilters, setShowFilters] = useState(false);
  const [namespaces, setNamespaces] = useState<string[]>([]);
  const limit = 20;
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch available namespaces on mount
  useEffect(() => {
    const fetchNamespaces = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/archival/stats`);
        if (res.ok) {
          const stats = await res.json();
          setNamespaces(stats.map((s: { namespace: string }) => s.namespace));
        }
      } catch {
        // Non-critical, ignore
      }
    };
    fetchNamespaces();
  }, []);

  const performSearch = useCallback(
    async (searchOffset: number, append: boolean) => {
      if (!query.trim()) return;
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({
          query: query.trim(),
          limit: String(limit),
          offset: String(searchOffset),
        });
        if (namespace !== "all") params.set("namespace", namespace);

        const res = await fetch(`${API_BASE}/api/memory/search?${params}`);
        if (!res.ok) {
          throw new Error(`Search failed (${res.status})`);
        }
        const data: SearchResponse = await res.json();

        // Client-side sort for now (server may not support sort params)
        const sorted = [...data.results].sort((a, b) => {
          let cmp = 0;
          if (sortField === "relevance") cmp = a.cosine_score - b.cosine_score;
          else if (sortField === "importance") cmp = a.importance_score - b.importance_score;
          else if (sortField === "date") {
            const da = a.created_at ? new Date(a.created_at).getTime() : 0;
            const db = b.created_at ? new Date(b.created_at).getTime() : 0;
            cmp = da - db;
          }
          return sortDir === "desc" ? -cmp : cmp;
        });

        setResults((prev) => (append ? [...prev, ...sorted] : sorted));
        setTotal(data.total);
        setHasSearched(true);
      } catch (err: any) {
        setError(err.message || "Search failed");
      }
      setLoading(false);
    },
    [query, namespace, sortField, sortDir, limit]
  );

  const handleSearch = useCallback(() => {
    setOffset(0);
    setResults([]);
    performSearch(0, false);
  }, [performSearch]);

  const handleLoadMore = useCallback(() => {
    const newOffset = offset + limit;
    setOffset(newOffset);
    performSearch(newOffset, true);
  }, [offset, performSearch]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") handleSearch();
    },
    [handleSearch]
  );

  // Re-sort results when sort changes (without re-fetching)
  useEffect(() => {
    if (results.length === 0) return;
    setResults((prev) => {
      const sorted = [...prev].sort((a, b) => {
        let cmp = 0;
        if (sortField === "relevance") cmp = a.cosine_score - b.cosine_score;
        else if (sortField === "importance") cmp = a.importance_score - b.importance_score;
        else if (sortField === "date") {
          const da = a.created_at ? new Date(a.created_at).getTime() : 0;
          const db = b.created_at ? new Date(b.created_at).getTime() : 0;
          cmp = da - db;
        }
        return sortDir === "desc" ? -cmp : cmp;
      });
      return sorted;
    });
  }, [sortField, sortDir]);

  const hasMore = results.length < total;

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDir === "desc" ? (
      <SortDesc className="w-3 h-3 ml-0.5 inline" />
    ) : (
      <SortAsc className="w-3 h-3 ml-0.5 inline" />
    );
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
            <Search className="w-6 h-6 text-neutral-400" />
            <h1 className="text-2xl font-semibold">Memory Search</h1>
          </div>
        </div>

        {/* Search Bar */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-500" />
            <Input
              ref={inputRef}
              placeholder="Search memories..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              className="pl-9 bg-neutral-700 border-neutral-600 text-neutral-100 placeholder:text-neutral-500"
            />
          </div>
          <Button
            variant="outline"
            size="icon"
            className="border-neutral-600 text-neutral-400 hover:text-neutral-100"
            onClick={() => setShowFilters(!showFilters)}
          >
            <SlidersHorizontal className="w-4 h-4" />
          </Button>
          <Button
            className="bg-sky-600 hover:bg-sky-700 text-white"
            onClick={handleSearch}
            disabled={loading || !query.trim()}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Search"}
          </Button>
        </div>

        {/* Filters */}
        {showFilters && (
          <div className="flex items-center gap-4 p-3 bg-neutral-700 rounded-lg border border-neutral-600">
            <span className="text-xs text-neutral-400">Namespace:</span>
            <Select value={namespace} onValueChange={setNamespace}>
              <SelectTrigger className="w-48 bg-neutral-800 border-neutral-600 text-neutral-200 h-8 text-xs">
                <SelectValue placeholder="All namespaces" />
              </SelectTrigger>
              <SelectContent className="bg-neutral-800 border-neutral-600">
                <SelectItem value="all" className="text-neutral-200">All Namespaces</SelectItem>
                {namespaces.map((ns) => (
                  <SelectItem key={ns} value={ns} className="text-neutral-200">
                    {ns}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <span className="text-xs text-neutral-400 ml-4">Sort by:</span>
            <div className="flex gap-1">
              {(["relevance", "date", "importance"] as SortField[]).map((field) => (
                <Button
                  key={field}
                  variant="ghost"
                  size="sm"
                  className={`text-xs h-7 px-2 ${
                    sortField === field ? "text-sky-400 bg-neutral-600" : "text-neutral-400"
                  }`}
                  onClick={() => toggleSort(field)}
                >
                  {field.charAt(0).toUpperCase() + field.slice(1)}
                  <SortIcon field={field} />
                </Button>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="text-sm text-red-400 bg-red-900/20 border border-red-800 rounded p-3">
            {error}
          </div>
        )}

        {/* Results Summary */}
        {hasSearched && !loading && (
          <div className="text-xs text-neutral-500">
            {results.length} of {total} results for &ldquo;{query}&rdquo;
          </div>
        )}

        {/* Results */}
        <div className="flex-1 min-h-0 overflow-auto space-y-2">
          {!hasSearched && !loading && (
            <div className="flex flex-col items-center justify-center h-64 text-neutral-500">
              <Search className="w-12 h-12 mb-4 opacity-30" />
              <p className="text-lg">Search your memories</p>
              <p className="text-sm">Query across archival memory to find relevant stored knowledge</p>
            </div>
          )}

          {loading && results.length === 0 && (
            <div className="flex justify-center py-12">
              <Loader2 className="w-6 h-6 text-neutral-500 animate-spin" />
            </div>
          )}

          {results.map((result) => (
            <Card
              key={result.id}
              className="bg-neutral-700 border-neutral-600 cursor-pointer hover:border-neutral-500 transition-colors"
              onClick={() => setExpandedId(expandedId === result.id ? null : result.id)}
            >
              <CardContent className="p-3">
                {/* Header row */}
                <div className="flex items-center gap-2 mb-2">
                  <Badge className={`text-xs ${NAMESPACE_COLORS[result.namespace] || "bg-neutral-600"}`}>
                    {result.namespace}
                  </Badge>

                  {/* Importance indicator */}
                  <div className="flex items-center gap-1.5">
                    <div className="w-16 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${getImportanceColor(result.importance_score)}`}
                        style={{ width: `${result.importance_score * 100}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-neutral-500">
                      {(result.importance_score * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Similarity */}
                  <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-neutral-600 text-neutral-400">
                    sim: {result.cosine_score.toFixed(3)}
                  </Badge>

                  <span className="text-xs text-neutral-500 ml-auto">
                    {timeAgo(result.created_at)}
                  </span>

                  {expandedId === result.id ? (
                    <ChevronUp className="w-4 h-4 text-neutral-400" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-neutral-400" />
                  )}
                </div>

                {/* Content */}
                <p className="text-sm text-neutral-200">
                  {expandedId === result.id
                    ? result.content
                    : truncate(result.content, 200)}
                </p>

                {/* Expanded metadata */}
                {expandedId === result.id && (
                  <div className="mt-3 pt-3 border-t border-neutral-600 space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-neutral-500">
                        ID: <span className="text-neutral-400 font-mono">{result.id}</span>
                      </div>
                      <div className="text-neutral-500">
                        Importance: <span className="text-neutral-400">{result.importance_score.toFixed(3)}</span>
                      </div>
                      <div className="text-neutral-500">
                        Cosine Similarity: <span className="text-neutral-400">{result.cosine_score.toFixed(4)}</span>
                      </div>
                      <div className="text-neutral-500">
                        Weighted Score: <span className="text-neutral-400">{result.score.toFixed(4)}</span>
                      </div>
                      <div className="text-neutral-500">
                        Created: <span className="text-neutral-400">{result.created_at || "N/A"}</span>
                      </div>
                    </div>
                    {Object.keys(result.metadata || {}).length > 0 && (
                      <div>
                        <p className="text-xs text-neutral-500 mb-1">Metadata:</p>
                        <pre className="text-xs text-neutral-400 bg-neutral-800 rounded p-2 whitespace-pre-wrap break-all max-h-40 overflow-auto">
                          {JSON.stringify(result.metadata, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}

          {/* Load More */}
          {hasMore && hasSearched && (
            <div className="flex justify-center py-4">
              <Button
                variant="outline"
                className="border-neutral-600 text-neutral-300 hover:text-neutral-100"
                onClick={handleLoadMore}
                disabled={loading}
              >
                {loading ? (
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                ) : null}
                Load More ({results.length} of {total})
              </Button>
            </div>
          )}

          {/* No results */}
          {hasSearched && !loading && results.length === 0 && (
            <div className="text-center py-12 text-neutral-500">
              <Search className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p>No memories found matching &ldquo;{query}&rdquo;</p>
              <p className="text-xs mt-1">Try different keywords or broader search terms</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
