import { useState, useCallback, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft,
  Search,
  Loader2,
  Network,
  X,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

// ---- Types ----
interface GraphNode {
  id: string;
  name: string;
  type: string;
  mention_count: number;
  memories: RelatedMemory[];
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

interface RelatedMemory {
  id: string;
  content: string;
  namespace: string;
  created_at: string;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// Force-directed layout node with physics state
interface SimNode extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

// ---- Force simulation constants ----
const REPULSION = 800;
const ATTRACTION = 0.005;
const DAMPING = 0.85;
const CENTER_GRAVITY = 0.01;
const ITERATIONS_PER_FRAME = 3;

export function KnowledgeGraphView() {
  const [searchEntity, setSearchEntity] = useState("");
  const [radius, setRadius] = useState(2);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [simNodes, setSimNodes] = useState<SimNode[]>([]);
  const [simEdges, setSimEdges] = useState<GraphEdge[]>([]);
  const animRef = useRef<number>(0);
  const svgRef = useRef<SVGSVGElement>(null);
  const draggingRef = useRef<{
    id: string;
    offsetX: number;
    offsetY: number;
  } | null>(null);
  const [svgSize, setSvgSize] = useState({ width: 800, height: 600 });

  // Measure SVG container
  useEffect(() => {
    function measure() {
      if (svgRef.current) {
        const rect = svgRef.current.getBoundingClientRect();
        setSvgSize({ width: rect.width || 800, height: rect.height || 600 });
      }
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  // Fetch graph data
  const fetchGraph = useCallback(async () => {
    if (!searchEntity.trim()) return;
    setLoading(true);
    setError(null);
    setSelectedNode(null);

    try {
      const params = new URLSearchParams({
        center: searchEntity.trim(),
        radius: String(radius),
      });
      const res = await fetch(`${API_BASE}/api/analytics/knowledge-graph?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: GraphData = await res.json();

      const nodes = data.nodes ?? [];
      const edges = data.edges ?? [];

      if (nodes.length === 0) {
        setGraphData({ nodes: [], edges: [] });
        setSimNodes([]);
        setSimEdges([]);
        setLoading(false);
        return;
      }

      // Initialise simulation positions
      const cx = svgSize.width / 2;
      const cy = svgSize.height / 2;
      const maxMentions = Math.max(...nodes.map((n) => n.mention_count), 1);
      const sn: SimNode[] = nodes.map((n, i) => {
        const angle = (2 * Math.PI * i) / nodes.length;
        const spread = Math.min(svgSize.width, svgSize.height) * 0.3;
        return {
          ...n,
          x: cx + Math.cos(angle) * spread + (Math.random() - 0.5) * 40,
          y: cy + Math.sin(angle) * spread + (Math.random() - 0.5) * 40,
          vx: 0,
          vy: 0,
          radius: 12 + (n.mention_count / maxMentions) * 28,
        };
      });

      setGraphData({ nodes, edges });
      setSimNodes(sn);
      setSimEdges(edges);
    } catch (err: any) {
      setError(err.message ?? "Failed to fetch graph");
    } finally {
      setLoading(false);
    }
  }, [searchEntity, radius, svgSize.width, svgSize.height]);

  // Force simulation loop
  useEffect(() => {
    if (simNodes.length === 0) return;
    let running = true;

    function tick() {
      if (!running) return;
      setSimNodes((prev) => {
        if (prev.length === 0) return prev;
        const next = prev.map((n) => ({ ...n }));
        const cx = svgSize.width / 2;
        const cy = svgSize.height / 2;
        const pad = 40;

        for (let iter = 0; iter < ITERATIONS_PER_FRAME; iter++) {
          // Repulsion between all pairs
          for (let i = 0; i < next.length; i++) {
            for (let j = i + 1; j < next.length; j++) {
              const dx = next[i].x - next[j].x;
              const dy = next[i].y - next[j].y;
              const distSq = dx * dx + dy * dy + 1;
              const dist = Math.sqrt(distSq);
              const force = REPULSION / distSq;
              const fx = (dx / dist) * force;
              const fy = (dy / dist) * force;
              next[i].vx += fx;
              next[i].vy += fy;
              next[j].vx -= fx;
              next[j].vy -= fy;
            }
          }

          // Attraction along edges
          const nodeMap = new Map(next.map((n) => [n.id, n]));
          for (const edge of simEdges) {
            const a = nodeMap.get(edge.source);
            const b = nodeMap.get(edge.target);
            if (!a || !b) continue;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dist = Math.sqrt(dx * dx + dy * dy) + 1;
            const force = ATTRACTION * dist * edge.weight;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            a.vx += fx;
            a.vy += fy;
            b.vx -= fx;
            b.vy -= fy;
          }

          // Center gravity + damping + position update
          for (const n of next) {
            if (draggingRef.current?.id === n.id) continue;
            n.vx += (cx - n.x) * CENTER_GRAVITY;
            n.vy += (cy - n.y) * CENTER_GRAVITY;
            n.vx *= DAMPING;
            n.vy *= DAMPING;
            n.x += n.vx;
            n.y += n.vy;
            // Bounds
            n.x = Math.max(pad, Math.min(svgSize.width - pad, n.x));
            n.y = Math.max(pad, Math.min(svgSize.height - pad, n.y));
          }
        }
        return next;
      });
      animRef.current = requestAnimationFrame(tick);
    }
    animRef.current = requestAnimationFrame(tick);
    return () => {
      running = false;
      cancelAnimationFrame(animRef.current);
    };
  }, [simNodes.length, simEdges, svgSize.width, svgSize.height]);

  // Drag handlers
  const handleMouseDown = useCallback(
    (id: string, e: React.MouseEvent) => {
      const node = simNodes.find((n) => n.id === id);
      if (!node) return;
      const svg = svgRef.current;
      if (!svg) return;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const svgPt = pt.matrixTransform(svg.getScreenCTM()!.inverse());
      draggingRef.current = {
        id,
        offsetX: node.x - svgPt.x,
        offsetY: node.y - svgPt.y,
      };
    },
    [simNodes]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!draggingRef.current) return;
      const svg = svgRef.current;
      if (!svg) return;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const svgPt = pt.matrixTransform(svg.getScreenCTM()!.inverse());
      const { id, offsetX, offsetY } = draggingRef.current;
      setSimNodes((prev) =>
        prev.map((n) =>
          n.id === id
            ? { ...n, x: svgPt.x + offsetX, y: svgPt.y + offsetY, vx: 0, vy: 0 }
            : n
        )
      );
    },
    []
  );

  const handleMouseUp = useCallback(() => {
    draggingRef.current = null;
  }, []);

  // Edge weight normalisation
  const maxEdgeWeight = simEdges.reduce((m, e) => Math.max(m, e.weight), 1);

  // Colour by node type
  const nodeColor = (type: string) => {
    const colors: Record<string, string> = {
      person: "#a78bfa", // purple
      organization: "#38bdf8", // sky
      location: "#4ade80", // green
      concept: "#fbbf24", // amber
      event: "#f87171", // red
    };
    return colors[type?.toLowerCase()] ?? "#94a3b8";
  };

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      fetchGraph();
    },
    [fetchGraph]
  );

  return (
    <div className="flex justify-center w-full h-screen bg-neutral-800 text-neutral-100 overflow-hidden">
      <div className="flex flex-col w-full max-w-[1280px] p-6 gap-4 h-full">
        {/* Header */}
        <div className="flex items-center justify-between shrink-0">
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
            <Network className="w-6 h-6 text-neutral-400" />
            <h1 className="text-2xl font-semibold">Knowledge Graph</h1>
          </div>
        </div>

        {/* Controls */}
        <form
          onSubmit={handleSubmit}
          className="flex flex-col sm:flex-row gap-3 shrink-0"
        >
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-500" />
            <Input
              value={searchEntity}
              onChange={(e) => setSearchEntity(e.target.value)}
              placeholder="Search for an entity (e.g. person, concept)..."
              className="pl-9 bg-neutral-700 border-neutral-600 text-neutral-100 placeholder:text-neutral-500"
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-400 whitespace-nowrap">
              Radius: {radius}
            </span>
            <input
              type="range"
              min={1}
              max={3}
              value={radius}
              onChange={(e) => setRadius(Number(e.target.value))}
              className="w-24 accent-sky-500"
            />
          </div>

          <Button
            type="submit"
            disabled={loading || !searchEntity.trim()}
            className="bg-sky-600 hover:bg-sky-700 text-white"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Network className="w-4 h-4" />
            )}
            <span className="ml-1">Explore</span>
          </Button>
        </form>

        {/* Main content: graph + sidebar */}
        <div className="flex flex-1 gap-4 min-h-0">
          {/* Graph area */}
          <div className="flex-1 bg-neutral-900 rounded-lg border border-neutral-700 overflow-hidden relative">
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center z-10 bg-neutral-900/80">
                <Loader2 className="w-8 h-8 text-sky-400 animate-spin" />
              </div>
            )}

            {error && (
              <div className="absolute inset-0 flex items-center justify-center z-10">
                <Card className="bg-red-900/30 border-red-800">
                  <CardContent className="p-4 text-red-300 text-sm">
                    {error}
                  </CardContent>
                </Card>
              </div>
            )}

            {!loading && !error && graphData && graphData.nodes.length === 0 && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-neutral-500 gap-2">
                <Network className="w-10 h-10 opacity-20" />
                <p className="text-sm">No entities found for this query.</p>
              </div>
            )}

            {!loading && !error && !graphData && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-neutral-500 gap-2">
                <Network className="w-12 h-12 opacity-20" />
                <p className="text-sm">
                  Search for an entity to visualise the knowledge graph.
                </p>
              </div>
            )}

            <svg
              ref={svgRef}
              width="100%"
              height="100%"
              className="cursor-grab active:cursor-grabbing"
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <defs>
                <marker
                  id="arrow"
                  viewBox="0 0 10 10"
                  refX="10"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#525252" />
                </marker>
              </defs>

              {/* Edges */}
              {simNodes.length > 0 &&
                simEdges.map((edge, i) => {
                  const src = simNodes.find((n) => n.id === edge.source);
                  const tgt = simNodes.find((n) => n.id === edge.target);
                  if (!src || !tgt) return null;
                  const opacity = 0.15 + (edge.weight / maxEdgeWeight) * 0.5;
                  const strokeWidth = 1 + (edge.weight / maxEdgeWeight) * 3;
                  return (
                    <line
                      key={i}
                      x1={src.x}
                      y1={src.y}
                      x2={tgt.x}
                      y2={tgt.y}
                      stroke="#525252"
                      strokeWidth={strokeWidth}
                      opacity={opacity}
                      markerEnd="url(#arrow)"
                    />
                  );
                })}

              {/* Nodes */}
              {simNodes.map((node) => {
                const isSelected = selectedNode?.id === node.id;
                const color = nodeColor(node.type);
                return (
                  <g
                    key={node.id}
                    onMouseDown={(e) => handleMouseDown(node.id, e)}
                    onClick={() => setSelectedNode(node)}
                    className="cursor-pointer"
                  >
                    {isSelected && (
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={node.radius + 4}
                        fill="none"
                        stroke={color}
                        strokeWidth={2}
                        opacity={0.5}
                      />
                    )}
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={node.radius}
                      fill={color}
                      opacity={isSelected ? 1 : 0.75}
                      stroke={isSelected ? "#fff" : "none"}
                      strokeWidth={isSelected ? 2 : 0}
                    />
                    <text
                      x={node.x}
                      y={node.y + node.radius + 14}
                      textAnchor="middle"
                      fill="#d4d4d4"
                      fontSize={11}
                      fontFamily="sans-serif"
                      className="pointer-events-none select-none"
                    >
                      {node.name.length > 16
                        ? node.name.slice(0, 14) + "..."
                        : node.name}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Sidebar */}
          <div
            className={cn(
              "bg-neutral-900 rounded-lg border border-neutral-700 overflow-y-auto transition-all duration-200",
              selectedNode ? "w-80" : "w-0 opacity-0"
            )}
          >
            {selectedNode && (
              <div className="p-4 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-neutral-100 text-lg">
                    {selectedNode.name}
                  </h3>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-neutral-500 hover:text-neutral-200"
                    onClick={() => setSelectedNode(null)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>

                <div className="flex items-center gap-2">
                  <Badge
                    className="text-[10px] border-0"
                    style={{
                      backgroundColor: nodeColor(selectedNode.type),
                      color: "#000",
                    }}
                  >
                    {selectedNode.type}
                  </Badge>
                  <span className="text-xs text-neutral-500">
                    {selectedNode.mention_count} mention
                    {selectedNode.mention_count !== 1 ? "s" : ""}
                  </span>
                </div>

                <h4 className="text-xs font-medium text-neutral-400 uppercase tracking-wider">
                  Related Memories
                </h4>

                {selectedNode.memories && selectedNode.memories.length > 0 ? (
                  <div className="flex flex-col gap-2">
                    {selectedNode.memories.map((mem) => (
                      <Card
                        key={mem.id}
                        className="bg-neutral-800 border-neutral-600"
                      >
                        <CardContent className="p-3">
                          <p className="text-xs text-neutral-300 line-clamp-4 whitespace-pre-wrap">
                            {mem.content}
                          </p>
                          <div className="flex items-center justify-between mt-2">
                            <Badge
                              variant="outline"
                              className="text-[9px] border-neutral-600 text-neutral-400"
                            >
                              {mem.namespace}
                            </Badge>
                            <span className="text-[10px] text-neutral-600">
                              {new Date(mem.created_at).toLocaleDateString()}
                            </span>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-neutral-600">
                    No related memories found.
                  </p>
                )}

                {/* Connected nodes */}
                {simEdges.filter(
                  (e) =>
                    e.source === selectedNode.id ||
                    e.target === selectedNode.id
                ).length > 0 && (
                  <>
                    <h4 className="text-xs font-medium text-neutral-400 uppercase tracking-wider mt-2">
                      Connections
                    </h4>
                    <div className="flex flex-col gap-1">
                      {simEdges
                        .filter(
                          (e) =>
                            e.source === selectedNode.id ||
                            e.target === selectedNode.id
                        )
                        .map((edge, i) => {
                          const otherId =
                            edge.source === selectedNode.id
                              ? edge.target
                              : edge.source;
                          const other = graphData?.nodes.find(
                            (n) => n.id === otherId
                          );
                          if (!other) return null;
                          return (
                            <button
                              key={i}
                              onClick={() => setSelectedNode(other)}
                              className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-neutral-800 text-left"
                            >
                              <ChevronRight className="w-3 h-3 text-neutral-600" />
                              <span
                                className="w-2 h-2 rounded-full"
                                style={{
                                  backgroundColor: nodeColor(other.type),
                                }}
                              />
                              <span className="text-xs text-neutral-300 truncate">
                                {other.name}
                              </span>
                              <span className="text-[10px] text-neutral-600 ml-auto">
                                w:{edge.weight}
                              </span>
                            </button>
                          );
                        })}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
