import { useState, useCallback, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ArrowLeft,
  Search,
  Loader2,
  ChevronDown,
  ChevronUp,
  SlidersHorizontal,
  Brain,
} from "lucide-react";
import { cn } from "@/lib/utils";

const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

interface MemoryResult {
  id: string;
  content: string;
  namespace: string;
  importance: number;
  created_at: string;
  similarity: number;
}

interface SearchResponse {
  results: MemoryResult[];
  total: number;
}

type SortKey = "relevance" | "date" | "importance";

export function MemorySearchPanel() {
  const [query, setQuery] = useState("");
  const [namespace, setNamespace] = useState<string>("all");
  const [namespaces, setNamespaces] = useState<string[]>([]);
  const [results, setResults] = useState<MemoryResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<SortKey>("relevance");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [hasSearched, setHasSearched] = useState(false);
  const limit = 20;
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch available namespaces on mount
  useEffect(() => {
    async function fetchNamespaces() {
      try {
        const res = await fetch(`${API_BASE}/api/analytics/stats`);
        if (res.ok) {
          const data = await res.json();
          const ns: string[] =
            data.namespaces?.map((n: { namespace: string }) => n.namespace) ??
            [];
          setNamespaces(ns);
        }
      } catch {
        // Namespaces list is optional; the UI still works with "all"
      }
    }
    fetchNamespaces();
  }, []);

  const doSearch = useCallback(
    async (offset = 0) => {
      if (!query.trim()) return;
      const isLoadMore = offset > 0;
      if (isLoadMore) setLoadingMore(true);
      else setLoading(true);
      setError(null);

      try {
        const params = new URLSearchParams({
          query: query.trim(),
          limit: String(limit),
          offset: String(offset),
        });
        if (namespace !== "all") params.set("namespace", namespace);

        const res = await fetch(`${API_BASE}/api/memory/search?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: SearchResponse = await res.json();

        if (isLoadMore) {
          setResults((prev) => [...prev, ...(data.results ?? [])]);
        } else {
          setResults(data.results ?? []);
          setHasSearched(true);
        }
        setTotal(data.total ?? 0);
      } catch (err: any) {
        setError(err.message ?? "Search failed");
      } finally {
        setLoading(false);
        setLoadingMore(false);
      }
    },
    [query, namespace]
  );

  const handleSearch = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      doSearch(0);
    },
    [doSearch]
  );

  const handleLoadMore = useCallback(() => {
    doSearch(results.length);
  }, [doSearch, results.length]);

  const toggleExpand = useCallback((id: string) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  }, []);

  // Sort results client-side
  const sortedResults = [...results].sort((a, b) => {
    if (sortBy === "date") {
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    }
    if (sortBy === "importance") return b.importance - a.importance;
    return b.similarity - a.similarity; // relevance (default)
  });

  const formatDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return iso;
    }
  };

  const nsBadgeColor = (ns: string) => {
    const palette: Record<string, string> = {
      default: "bg-sky-700 text-sky-100",
      personal: "bg-purple-700 text-purple-100",
      professional: "bg-emerald-700 text-emerald-100",
      factual: "bg-amber-700 text-amber-100",
      context: "bg-rose-700 text-rose-100",
    };
    return palette[ns] ?? "bg-neutral-600 text-neutral-200";
  };

  return (
    <div className="flex justify-center w-full h-screen bg-neutral-800 text-neutral-100 overflow-auto">
      <div className="flex flex-col w-[1024px] max-w-[calc(100vw-3rem)] p-6 gap-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/">
              <Button
                variant="ghost"
                size="icon"
                className="text-neutral-400 hover:text-neutral-100"
              >
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <Brain className="w-6 h-6 text-neutral-400" />
            <h1 className="text-2xl font-semibold">Memory Search</h1>
          </div>
        </div>

        {/* Search bar */}
        <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-500" />
            <Input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search memories..."
              className="pl-9 bg-neutral-700 border-neutral-600 text-neutral-100 placeholder:text-neutral-500"
            />
          </div>

          <Select value={namespace} onValueChange={setNamespace}>
            <SelectTrigger className="w-[180px] bg-neutral-700 border-neutral-600 text-neutral-200">
              <SelectValue placeholder="All namespaces" />
            </SelectTrigger>
            <SelectContent className="bg-neutral-800 border-neutral-600">
              <SelectItem value="all">All namespaces</SelectItem>
              {namespaces.map((ns) => (
                <SelectItem key={ns} value={ns}>
                  {ns}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            type="submit"
            disabled={loading || !query.trim()}
            className="bg-sky-600 hover:bg-sky-700 text-white"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            <span className="ml-1">Search</span>
          </Button>
        </form>

        {/* Sort + results count */}
        {hasSearched && !loading && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-neutral-400">
              {total} result{total !== 1 ? "s" : ""} found
            </span>
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="w-3.5 h-3.5 text-neutral-500" />
              <Select
                value={sortBy}
                onValueChange={(v) => setSortBy(v as SortKey)}
              >
                <SelectTrigger className="w-[140px] h-8 text-xs bg-neutral-700 border-neutral-600 text-neutral-300">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-neutral-800 border-neutral-600">
                  <SelectItem value="relevance">Relevance</SelectItem>
                  <SelectItem value="date">Date</SelectItem>
                  <SelectItem value="importance">Importance</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20 gap-3">
            <Loader2 className="w-8 h-8 text-sky-400 animate-spin" />
            <span className="text-sm text-neutral-500">Searching memories...</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <Card className="bg-red-900/30 border-red-800">
            <CardContent className="p-4 text-red-300 text-sm">{error}</CardContent>
          </Card>
        )}

        {/* Results */}
        {!loading && sortedResults.length > 0 && (
          <div className="flex flex-col gap-3">
            {sortedResults.map((r) => {
              const isExpanded = !!expanded[r.id];
              const isLong = r.content.length > 200;
              const displayContent =
                isExpanded || !isLong
                  ? r.content
                  : r.content.slice(0, 200) + "...";

              return (
                <Card
                  key={r.id}
                  className="bg-neutral-700 border-neutral-600 hover:border-neutral-500 transition-colors"
                >
                  <CardContent className="p-4 flex flex-col gap-3">
                    {/* Top row: namespace + similarity */}
                    <div className="flex items-center justify-between">
                      <Badge
                        className={cn(
                          "text-[10px] font-medium border-0",
                          nsBadgeColor(r.namespace)
                        )}
                      >
                        {r.namespace}
                      </Badge>
                      <span className="text-xs text-neutral-500">
                        {(r.similarity * 100).toFixed(1)}% match
                      </span>
                    </div>

                    {/* Content */}
                    <p className="text-sm text-neutral-200 whitespace-pre-wrap leading-relaxed">
                      {displayContent}
                    </p>
                    {isLong && (
                      <button
                        onClick={() => toggleExpand(r.id)}
                        className="flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300 self-start"
                      >
                        {isExpanded ? (
                          <>
                            <ChevronUp className="w-3 h-3" /> Show less
                          </>
                        ) : (
                          <>
                            <ChevronDown className="w-3 h-3" /> Show more
                          </>
                        )}
                      </button>
                    )}

                    {/* Bottom row: importance bar + date */}
                    <div className="flex items-center gap-4 mt-1">
                      <div className="flex items-center gap-2 flex-1">
                        <span className="text-[10px] text-neutral-500 w-16">
                          Importance
                        </span>
                        <div className="flex-1 bg-neutral-800 rounded-full h-1.5 max-w-[120px]">
                          <div
                            className={cn(
                              "h-1.5 rounded-full",
                              r.importance > 0.7
                                ? "bg-emerald-500"
                                : r.importance > 0.4
                                ? "bg-amber-500"
                                : "bg-neutral-500"
                            )}
                            style={{
                              width: `${Math.round(r.importance * 100)}%`,
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-neutral-500 w-8">
                          {Math.round(r.importance * 100)}%
                        </span>
                      </div>
                      <span className="text-xs text-neutral-500">
                        {formatDate(r.created_at)}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              );
            })}

            {/* Load more */}
            {results.length < total && (
              <Button
                variant="outline"
                onClick={handleLoadMore}
                disabled={loadingMore}
                className="self-center border-neutral-600 text-neutral-300 hover:bg-neutral-700"
              >
                {loadingMore ? (
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                ) : null}
                Load more ({results.length} / {total})
              </Button>
            )}
          </div>
        )}

        {/* Empty state after search */}
        {!loading && hasSearched && sortedResults.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center py-20 gap-3 text-neutral-500">
            <Search className="w-10 h-10 opacity-30" />
            <p className="text-sm">No memories found for this query.</p>
          </div>
        )}

        {/* Initial state */}
        {!loading && !hasSearched && (
          <div className="flex flex-col items-center justify-center py-20 gap-3 text-neutral-500">
            <Brain className="w-12 h-12 opacity-20" />
            <p className="text-sm">Enter a query to search your memories.</p>
          </div>
        )}
      </div>
    </div>
  );
}
