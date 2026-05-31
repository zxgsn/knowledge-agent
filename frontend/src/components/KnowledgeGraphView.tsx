import { useState, useCallback, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, GitBranch, Loader2, Search, Minus, Plus, RotateCcw } from "lucide-react";

const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

interface GraphNode {
  id: string;
  label: string;
  namespace: string;
  mention_count: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

const NAMESPACE_COLORS: Record<string, string> = {
  ingested: "#38bdf8",
  research: "#34d399",
  manual: "#fbbf24",
  conversation_facts: "#c084fc",
};

const DEFAULT_COLOR = "#a3a3a3";

function getNodeColor(namespace: string): string {
  return NAMESPACE_COLORS[namespace] || DEFAULT_COLOR;
}

// Simple force-directed layout simulation
function simulateStep(
  nodes: GraphNode[],
  edges: GraphEdge[],
  width: number,
  height: number
): boolean {
  const alpha = 0.3;
  const repulsion = 800;
  const attraction = 0.01;
  const damping = 0.85;
  const centerGravity = 0.01;

  let totalMovement = 0;

  // Center gravity
  const cx = width / 2;
  const cy = height / 2;
  for (const node of nodes) {
    if (node.x === undefined || node.y === undefined) continue;
    node.vx = (node.vx || 0) + (cx - node.x) * centerGravity;
    node.vy = (node.vy || 0) + (cy - node.y) * centerGravity;
  }

  // Repulsion between all node pairs
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      if (a.x === undefined || a.y === undefined || b.x === undefined || b.y === undefined) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const force = repulsion / (dist * dist);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.vx = (a.vx || 0) - fx;
      a.vy = (a.vy || 0) - fy;
      b.vx = (b.vx || 0) + fx;
      b.vy = (b.vy || 0) + fy;
    }
  }

  // Attraction along edges
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  for (const edge of edges) {
    const a = nodeMap.get(edge.source);
    const b = nodeMap.get(edge.target);
    if (!a || !b || a.x === undefined || a.y === undefined || b.x === undefined || b.y === undefined) continue;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 1) continue;
    const force = dist * attraction * (1 + edge.weight * 0.2);
    const fx = (dx / dist) * force;
    const fy = (dy / dist) * force;
    a.vx = (a.vx || 0) + fx;
    a.vy = (a.vy || 0) + fy;
    b.vx = (b.vx || 0) - fx;
    b.vy = (b.vy || 0) - fy;
  }

  // Apply velocities
  for (const node of nodes) {
    if (node.x === undefined || node.y === undefined) continue;
    node.vx = (node.vx || 0) * damping;
    node.vy = (node.vy || 0) * damping;
    const newX = node.x + (node.vx || 0) * alpha;
    const newY = node.y + (node.vy || 0) * alpha;
    // Keep within bounds
    const padding = 60;
    node.x = Math.max(padding, Math.min(width - padding, newX));
    node.y = Math.max(padding, Math.min(height - padding, newY));
    totalMovement += Math.abs(node.vx || 0) + Math.abs(node.vy || 0);
  }

  return totalMovement > 0.5; // true if still moving
}

export function KnowledgeGraphView() {
  const [centerEntity, setCenterEntity] = useState("");
  const [radius, setRadius] = useState(2);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [dragNode, setDragNode] = useState<string | null>(null);
  const [relatedMemories, setRelatedMemories] = useState<any[] | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<GraphNode[]>([]);
  const animRef = useRef<number>(0);
  const dragOffset = useRef({ x: 0, y: 0 });

  const fetchGraph = useCallback(async () => {
    if (!centerEntity.trim()) return;
    setLoading(true);
    setError(null);
    setRelatedMemories(null);
    setSelectedNode(null);
    try {
      const params = new URLSearchParams({
        center: centerEntity.trim(),
        radius: String(radius),
      });
      const res = await fetch(`${API_BASE}/api/analytics/knowledge-graph?${params}`);
      if (!res.ok) throw new Error(`Failed (${res.status})`);
      const data: GraphData = await res.json();

      // Initialize node positions
      const canvas = canvasRef.current;
      const w = canvas?.width || 800;
      const h = canvas?.height || 500;
      const existingMap = new Map(nodesRef.current.map((n) => [n.id, { x: n.x, y: n.y }]));

      for (const node of data.nodes) {
        const existing = existingMap.get(node.id);
        if (existing) {
          node.x = existing.x;
          node.y = existing.y;
        } else {
          node.x = w / 2 + (Math.random() - 0.5) * 200;
          node.y = h / 2 + (Math.random() - 0.5) * 200;
        }
        node.vx = 0;
        node.vy = 0;
      }

      nodesRef.current = data.nodes;
      setGraphData(data);
    } catch (err: any) {
      setError(err.message || "Failed to load graph");
    }
    setLoading(false);
  }, [centerEntity, radius]);

  const handleNodeClick = useCallback(
    async (nodeId: string) => {
      setSelectedNode(nodeId);
      try {
        const params = new URLSearchParams({ entity: nodeId, limit: "10" });
        const res = await fetch(`${API_BASE}/api/analytics/knowledge-graph/entity-memories?${params}`);
        if (res.ok) {
          setRelatedMemories(await res.json());
        } else {
          setRelatedMemories([]);
        }
      } catch {
        setRelatedMemories([]);
      }
    },
    []
  );

  // Canvas rendering loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resizeCanvas = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    };
    resizeCanvas();

    const nodes = nodesRef.current;
    const edges = graphData?.edges || [];
    let running = true;

    const draw = () => {
      if (!running) return;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      if (nodes.length === 0) {
        ctx.fillStyle = "#737373";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Search for an entity to visualize its knowledge graph", w / 2, h / 2);
        animRef.current = requestAnimationFrame(draw);
        return;
      }

      // Run simulation
      simulateStep(nodes, edges, w, h);

      const nodeMap = new Map(nodes.map((n) => [n.id, n]));

      // Draw edges
      for (const edge of edges) {
        const src = nodeMap.get(edge.source);
        const tgt = nodeMap.get(edge.target);
        if (!src || !tgt || src.x === undefined || src.y === undefined || tgt.x === undefined || tgt.y === undefined) continue;

        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.strokeStyle = `rgba(100, 100, 100, ${Math.min(0.3 + edge.weight * 0.1, 0.7)})`;
        ctx.lineWidth = Math.max(1, Math.min(edge.weight * 0.8, 4));
        ctx.stroke();
      }

      // Draw nodes
      const maxMentions = Math.max(...nodes.map((n) => n.mention_count), 1);
      for (const node of nodes) {
        if (node.x === undefined || node.y === undefined) continue;
        const baseRadius = 6;
        const radius = baseRadius + (node.mention_count / maxMentions) * 14;
        const color = getNodeColor(node.namespace);

        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = hoveredNode?.id === node.id ? "#ffffff" : color;
        ctx.fill();

        // Node border
        ctx.strokeStyle = hoveredNode?.id === node.id ? color : "rgba(0,0,0,0.3)";
        ctx.lineWidth = hoveredNode?.id === node.id ? 3 : 1;
        ctx.stroke();

        // Label
        ctx.fillStyle = "#d4d4d4";
        ctx.font = `${hoveredNode?.id === node.id ? "bold " : ""}11px sans-serif`;
        ctx.textAlign = "center";
        ctx.fillText(node.label, node.x, node.y + radius + 14);
      }

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);

    return () => {
      running = false;
      cancelAnimationFrame(animRef.current);
    };
  }, [graphData, hoveredNode]);

  // Mouse interaction handlers
  const getNodeAtPos = useCallback(
    (x: number, y: number): GraphNode | null => {
      const nodes = nodesRef.current;
      const maxMentions = Math.max(...nodes.map((n) => n.mention_count), 1);
      for (let i = nodes.length - 1; i >= 0; i--) {
        const node = nodes[i];
        if (node.x === undefined || node.y === undefined) continue;
        const baseRadius = 6;
        const r = baseRadius + (node.mention_count / maxMentions) * 14 + 4;
        const dx = x - node.x;
        const dy = y - node.y;
        if (dx * dx + dy * dy <= r * r) return node;
      }
      return null;
    },
    []
  );

  const getCanvasPos = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = getCanvasPos(e);
      setMousePos({ x: e.clientX, y: e.clientY });

      if (dragNode) {
        const node = nodesRef.current.find((n) => n.id === dragNode);
        if (node) {
          node.x = pos.x - dragOffset.current.x;
          node.y = pos.y - dragOffset.current.y;
          node.vx = 0;
          node.vy = 0;
        }
        return;
      }

      const node = getNodeAtPos(pos.x, pos.y);
      setHoveredNode(node);
      if (canvasRef.current) {
        canvasRef.current.style.cursor = node ? "pointer" : "default";
      }
    },
    [dragNode, getNodeAtPos, getCanvasPos]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = getCanvasPos(e);
      const node = getNodeAtPos(pos.x, pos.y);
      if (node && node.x !== undefined && node.y !== undefined) {
        setDragNode(node.id);
        dragOffset.current = { x: pos.x - node.x, y: pos.y - node.y };
      }
    },
    [getCanvasPos, getNodeAtPos]
  );

  const handleMouseUp = useCallback(() => {
    if (dragNode) {
      setDragNode(null);
    }
  }, [dragNode]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (dragNode) return; // Don't trigger click on drag end
      const pos = getCanvasPos(e);
      const node = getNodeAtPos(pos.x, pos.y);
      if (node) {
        handleNodeClick(node.id);
      }
    },
    [dragNode, getCanvasPos, getNodeAtPos, handleNodeClick]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") fetchGraph();
    },
    [fetchGraph]
  );

  return (
    <div className="flex flex-col h-screen bg-neutral-800 text-neutral-100">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-neutral-700">
        <Link to="/">
          <Button variant="ghost" size="icon" className="text-neutral-400 hover:text-neutral-100">
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </Link>
        <GitBranch className="w-6 h-6 text-neutral-400" />
        <h1 className="text-2xl font-semibold">Knowledge Graph</h1>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-neutral-700 bg-neutral-800/50">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-500" />
          <Input
            placeholder="Entity name (e.g. Python, Machine Learning)..."
            value={centerEntity}
            onChange={(e) => setCenterEntity(e.target.value)}
            onKeyDown={handleKeyDown}
            className="pl-9 bg-neutral-700 border-neutral-600 text-neutral-100 placeholder:text-neutral-500"
          />
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-400">Hops:</span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-neutral-400"
            onClick={() => setRadius((r) => Math.max(1, r - 1))}
          >
            <Minus className="w-3 h-3" />
          </Button>
          <span className="text-sm text-neutral-200 w-4 text-center">{radius}</span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-neutral-400"
            onClick={() => setRadius((r) => Math.min(3, r + 1))}
          >
            <Plus className="w-3 h-3" />
          </Button>
        </div>

        <Button
          variant="outline"
          size="icon"
          className="border-neutral-600 text-neutral-400"
          onClick={() => {
            nodesRef.current.forEach((n) => {
              if (canvasRef.current) {
                n.x = canvasRef.current.width / 2 + (Math.random() - 0.5) * 200;
                n.y = canvasRef.current.height / 2 + (Math.random() - 0.5) * 200;
                n.vx = 0;
                n.vy = 0;
              }
            });
          }}
          title="Reset positions"
        >
          <RotateCcw className="w-4 h-4" />
        </Button>

        <Button
          className="bg-sky-600 hover:bg-sky-700 text-white"
          onClick={fetchGraph}
          disabled={loading || !centerEntity.trim()}
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Visualize"}
        </Button>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-6 mt-3 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded p-3">
          {error}
        </div>
      )}

      {/* Canvas + Sidebar */}
      <div className="flex flex-1 min-h-0">
        {/* Canvas */}
        <div className="flex-1 relative">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onClick={handleClick}
          />

          {/* Tooltip */}
          {hoveredNode && !dragNode && (
            <div
              className="fixed z-50 pointer-events-none bg-neutral-900 border border-neutral-600 rounded px-3 py-2 shadow-lg text-xs"
              style={{ left: mousePos.x + 12, top: mousePos.y + 12 }}
            >
              <div className="flex items-center gap-2 mb-1">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: getNodeColor(hoveredNode.namespace) }}
                />
                <span className="font-medium text-neutral-100">{hoveredNode.label}</span>
              </div>
              <div className="text-neutral-400">
                <div>Namespace: {hoveredNode.namespace}</div>
                <div>Mentions: {hoveredNode.mention_count}</div>
              </div>
            </div>
          )}

          {/* Legend */}
          {graphData && graphData.nodes.length > 0 && (
            <div className="absolute bottom-4 left-4 bg-neutral-900/90 border border-neutral-700 rounded p-3 text-xs">
              <p className="text-neutral-400 mb-2 font-medium">Namespaces</p>
              {Object.entries(NAMESPACE_COLORS).map(([ns, color]) => (
                <div key={ns} className="flex items-center gap-2 mb-1">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                  <span className="text-neutral-300">{ns}</span>
                </div>
              ))}
              <p className="text-neutral-500 mt-2">Node size = mention count</p>
              <p className="text-neutral-500">Edge thickness = relationship weight</p>
              <p className="text-neutral-500">Drag nodes to reposition</p>
            </div>
          )}
        </div>

        {/* Related memories sidebar */}
        {selectedNode && (
          <div className="w-80 border-l border-neutral-700 bg-neutral-800/50 overflow-auto">
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-neutral-200">
                  Related Memories
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 text-xs text-neutral-400"
                  onClick={() => {
                    setSelectedNode(null);
                    setRelatedMemories(null);
                  }}
                >
                  Close
                </Button>
              </div>
              <p className="text-xs text-neutral-500 mb-3">
                Entity: <span className="text-neutral-300">{selectedNode}</span>
              </p>

              {relatedMemories === null ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="w-5 h-5 text-neutral-500 animate-spin" />
                </div>
              ) : relatedMemories.length === 0 ? (
                <p className="text-xs text-neutral-500">No related memories found.</p>
              ) : (
                <div className="space-y-2">
                  {relatedMemories.map((mem: any, i: number) => (
                    <Card key={i} className="bg-neutral-700 border-neutral-600">
                      <CardContent className="p-2">
                        <div className="flex items-center gap-1 mb-1">
                          <Badge className={`text-[10px] ${
                            mem.namespace === "research"
                              ? "bg-emerald-600"
                              : mem.namespace === "manual"
                                ? "bg-amber-600"
                                : mem.namespace === "ingested"
                                  ? "bg-sky-600"
                                  : "bg-neutral-600"
                          }`}>
                            {mem.namespace}
                          </Badge>
                          <span className="text-[10px] text-neutral-500 ml-auto">
                            {mem.importance ? (mem.importance * 100).toFixed(0) + "%" : ""}
                          </span>
                        </div>
                        <p className="text-xs text-neutral-200 line-clamp-4">
                          {mem.content}
                        </p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
