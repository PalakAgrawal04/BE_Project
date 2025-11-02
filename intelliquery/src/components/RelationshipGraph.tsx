import { useEffect, useRef } from "react";

interface Relationship {
  from: string;
  to: string;
  type: string;
}

interface RelationshipGraphProps {
  relationships: Relationship[];
}

const RelationshipGraph = ({ relationships }: RelationshipGraphProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = 400 * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = canvas.offsetWidth;
    const height = 400;

    // Extract unique nodes
    const nodeSet = new Set<string>();
    relationships.forEach((rel) => {
      nodeSet.add(rel.from);
      nodeSet.add(rel.to);
    });
    const nodes = Array.from(nodeSet);

    // Position nodes in a circle
    const nodePositions = new Map<string, { x: number; y: number }>();
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.35;

    nodes.forEach((node, idx) => {
      const angle = (idx / nodes.length) * 2 * Math.PI;
      nodePositions.set(node, {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      });
    });

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw relationships (edges)
    ctx.strokeStyle = "hsl(190 85% 45%)";
    ctx.lineWidth = 2;
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "hsl(215 15% 45%)";

    relationships.forEach((rel) => {
      const from = nodePositions.get(rel.from);
      const to = nodePositions.get(rel.to);
      if (!from || !to) return;

      // Draw line
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();

      // Draw label at midpoint
      const midX = (from.x + to.x) / 2;
      const midY = (from.y + to.y) / 2;
      ctx.fillText(rel.type, midX, midY);
    });

    // Draw nodes
    nodes.forEach((node) => {
      const pos = nodePositions.get(node);
      if (!pos) return;

      // Draw circle
      ctx.fillStyle = "hsl(215 85% 35%)";
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 30, 0, 2 * Math.PI);
      ctx.fill();

      // Draw border
      ctx.strokeStyle = "hsl(190 85% 45%)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw label
      ctx.fillStyle = "white";
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(node, pos.x, pos.y);
    });
  }, [relationships]);

  return (
    <div className="w-full bg-muted rounded-md overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: "400px" }}
      />
    </div>
  );
};

export default RelationshipGraph;
