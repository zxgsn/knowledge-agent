import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, BarChart3, Database, MessageSquare, FileText, AlertTriangle, History } from "lucide-react";

const API_BASE = import.meta.env.DEV
  ? "http://localhost:8000"
  : "";

interface AnalyticsStats {
  archival_count: number;
  namespace_count: number;
  namespaces: { namespace: string; count: number }[];
  recall_count: number;
  thread_count: number;
  document_count: number;
  pending_conflicts: number;
  version_count: number;
  age_distribution: {
    last_7_days: number;
    last_30_days: number;
    older: number;
  };
}

export function MemoryAnalytics() {
  const [stats, setStats] = useState<AnalyticsStats | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchStats = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/analytics/stats`);
      if (res.ok) setStats(await res.json());
    } catch (err) {
      console.error("Failed to fetch analytics:", err);
    }
    setLoading(false);
  }, []);

  useEffect(() => { fetchStats(); }, [fetchStats]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen bg-neutral-800 text-neutral-100">
        <div className="text-neutral-500">Loading analytics...</div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex justify-center items-center h-screen bg-neutral-800 text-neutral-100">
        <div className="text-red-400">Failed to load analytics.</div>
      </div>
    );
  }

  return (
    <div className="flex justify-center w-full h-screen bg-neutral-800 text-neutral-100 overflow-auto">
      <div className="flex flex-col w-[1024px] max-w-[calc(100vw-3rem)] p-6 gap-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/">
              <Button variant="ghost" size="icon" className="text-neutral-400 hover:text-neutral-100">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <BarChart3 className="w-6 h-6 text-neutral-400" />
            <h1 className="text-2xl font-semibold">Memory Analytics</h1>
          </div>
          <Button variant="ghost" size="icon" onClick={fetchStats} className="text-neutral-400 hover:text-neutral-100">
            <BarChart3 className="w-4 h-4" />
          </Button>
        </div>

        {/* Overview Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-neutral-700 border-neutral-600">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Database className="w-4 h-4 text-sky-400" />
                <span className="text-xs text-neutral-400">Archival Memory</span>
              </div>
              <div className="text-2xl font-bold text-neutral-100">{stats.archival_count}</div>
              <div className="text-xs text-neutral-500">{stats.namespace_count} namespaces</div>
            </CardContent>
          </Card>

          <Card className="bg-neutral-700 border-neutral-600">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <MessageSquare className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-neutral-400">Recall Memory</span>
              </div>
              <div className="text-2xl font-bold text-neutral-100">{stats.recall_count}</div>
              <div className="text-xs text-neutral-500">{stats.thread_count} threads</div>
            </CardContent>
          </Card>

          <Card className="bg-neutral-700 border-neutral-600">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <FileText className="w-4 h-4 text-amber-400" />
                <span className="text-xs text-neutral-400">Documents</span>
              </div>
              <div className="text-2xl font-bold text-neutral-100">{stats.document_count}</div>
            </CardContent>
          </Card>

          <Card className="bg-neutral-700 border-neutral-600">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <History className="w-4 h-4 text-purple-400" />
                <span className="text-xs text-neutral-400">Versions</span>
              </div>
              <div className="text-2xl font-bold text-neutral-100">{stats.version_count}</div>
              {stats.pending_conflicts > 0 && (
                <Badge className="mt-1 bg-amber-600 text-white text-[10px]">
                  {stats.pending_conflicts} conflicts
                </Badge>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Namespace Distribution */}
        <Card className="bg-neutral-700 border-neutral-600">
          <CardContent className="p-4">
            <h2 className="text-sm font-medium text-neutral-200 mb-3">Namespace Distribution</h2>
            <div className="space-y-2">
              {stats.namespaces.map((ns) => (
                <div key={ns.namespace} className="flex items-center gap-3">
                  <span className="text-xs text-neutral-400 w-32 truncate">{ns.namespace}</span>
                  <div className="flex-1 bg-neutral-800 rounded-full h-2">
                    <div
                      className="bg-sky-600 h-2 rounded-full"
                      style={{ width: `${(ns.count / stats.archival_count) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-neutral-500 w-12 text-right">{ns.count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Age Distribution */}
        <Card className="bg-neutral-700 border-neutral-600">
          <CardContent className="p-4">
            <h2 className="text-sm font-medium text-neutral-200 mb-3">Age Distribution</h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-emerald-400">{stats.age_distribution.last_7_days}</div>
                <div className="text-xs text-neutral-500">Last 7 days</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-amber-400">{stats.age_distribution.last_30_days}</div>
                <div className="text-xs text-neutral-500">Last 30 days</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-neutral-400">{stats.age_distribution.older}</div>
                <div className="text-xs text-neutral-500">Older</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
