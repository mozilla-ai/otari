import { useMemo } from "react";
import type { RouteExplanation } from "../router-sim";
import type { DemoData } from "../types";

const TASK_COLORS = ["#4a7d8f", "#c2680f", "#5b6e8c", "#1f6648", "#8a5a8f", "#9a6a3a"];

/** Force-directed kNN network of the prompts: nodes link to their nearest
 *  neighbors and a spring layout pulls similar prompts together. The query and
 *  the neighbors the router used are highlighted. */
export function KnnGraph({ demo, exp, k }: { demo: DemoData; exp: RouteExplanation; k: number }) {
  const items = demo.items;

  const { pos, edges } = useMemo(() => {
    const n = items.length;
    const unit = items.map((it) => {
      let s = 0;
      for (const v of it.embedding) s += v * v;
      const norm = Math.sqrt(s) || 1;
      return it.embedding.map((v) => v / norm);
    });
    const sim = (a: number[], bi: number) => {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += a[i] * unit[bi][i];
      return s;
    };
    // Undirected kNN edge set (each node to its k nearest, deduped).
    const seen = new Set<string>();
    const edges: [number, number][] = [];
    for (let i = 0; i < n; i++) {
      const nearest = items
        .map((_, j) => ({ j, s: j === i ? -Infinity : sim(unit[i], j) }))
        .sort((a, b) => b.s - a.s)
        .slice(0, k);
      for (const { j } of nearest) {
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (!seen.has(key)) {
          seen.add(key);
          edges.push([Math.min(i, j), Math.max(i, j)]);
        }
      }
    }
    // Deterministic spring layout (init on a ring, no randomness).
    const pos = items.map((_, i) => {
      const a = (2 * Math.PI * i) / n;
      return { x: Math.cos(a) * 0.8, y: Math.sin(a) * 0.8 };
    });
    const ITERS = 400;
    for (let t = 0; t < ITERS; t++) {
      const disp = pos.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dx = pos[i].x - pos[j].x;
          const dy = pos[i].y - pos[j].y;
          const d2 = dx * dx + dy * dy + 1e-4;
          const d = Math.sqrt(d2);
          const rep = 0.03 / d2;
          disp[i].x += (dx / d) * rep;
          disp[i].y += (dy / d) * rep;
          disp[j].x -= (dx / d) * rep;
          disp[j].y -= (dy / d) * rep;
        }
      }
      for (const [i, j] of edges) {
        const dx = pos[i].x - pos[j].x;
        const dy = pos[i].y - pos[j].y;
        const d = Math.sqrt(dx * dx + dy * dy) + 1e-4;
        const att = d * 0.04;
        disp[i].x -= (dx / d) * att;
        disp[i].y -= (dy / d) * att;
        disp[j].x += (dx / d) * att;
        disp[j].y += (dy / d) * att;
      }
      const cool = 0.85 * (1 - t / ITERS) + 0.05;
      for (let i = 0; i < n; i++) {
        disp[i].x -= pos[i].x * 0.01;
        disp[i].y -= pos[i].y * 0.01;
        pos[i].x += disp[i].x * cool;
        pos[i].y += disp[i].y * cool;
      }
    }
    return { pos, edges };
  }, [items, k]);

  const xs = pos.map((p) => p.x);
  const ys = pos.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const S = 360;
  const PAD = 22;
  const sx = (x: number) => PAD + (maxX === minX ? 0.5 : (x - minX) / (maxX - minX)) * (S - 2 * PAD);
  const sy = (y: number) => PAD + (maxY === minY ? 0.5 : (y - minY) / (maxY - minY)) * (S - 2 * PAD);

  const tasks = Array.from(new Set(items.map((it) => it.task)));
  const color = (tk: string) => TASK_COLORS[tasks.indexOf(tk) % TASK_COLORS.length];
  const qi = items.findIndex((it) => it.id === exp.item.id);
  const neighborIdx = new Set(exp.neighbors.map((nb) => items.findIndex((it) => it.id === nb.item.id)));
  // The request routes within one partition: only same-task prompts are searched.
  const part = exp.item.task;
  const inPart = (i: number) => items[i].task === part;
  const hotEdge = (i: number, j: number) => (i === qi && neighborIdx.has(j)) || (j === qi && neighborIdx.has(i));

  return (
    <div>
      <svg viewBox={`0 0 ${S} ${S}`} className="w-full rounded-xl border border-border bg-surface" role="img" aria-label="kNN network of prompts">
        {/* Only edges within the searched partition are drawn; other tasks are dimmed. */}
        {edges
          .filter(([i, j]) => inPart(i) && inPart(j))
          .map(([i, j]) => {
            const hot = hotEdge(i, j);
            return (
              <line
                key={`${i}-${j}`}
                x1={sx(pos[i].x)}
                y1={sy(pos[i].y)}
                x2={sx(pos[j].x)}
                y2={sy(pos[j].y)}
                stroke={hot ? "var(--accent)" : "var(--border)"}
                strokeWidth={hot ? 2 : 1}
                strokeOpacity={hot ? 0.85 : 0.5}
              />
            );
          })}
        {items.map((it, i) => {
          const isQ = i === qi;
          const isN = neighborIdx.has(i);
          const dim = !inPart(i);
          return (
            <circle
              key={it.id}
              cx={sx(pos[i].x)}
              cy={sy(pos[i].y)}
              r={isQ ? 7.5 : isN ? 5.5 : dim ? 3 : 4}
              fill={color(it.task)}
              fillOpacity={dim ? 0.12 : isQ || isN ? 1 : 0.55}
              stroke={isQ ? "var(--accent)" : isN ? "var(--foreground)" : "none"}
              strokeWidth={isQ ? 3 : isN ? 1.25 : 0}
            />
          );
        })}
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
        {tasks.map((t) => (
          <span key={t} className="inline-flex items-center gap-1">
            <span
              className="inline-block size-2.5 rounded-full"
              style={{ background: color(t), opacity: t === part ? 1 : 0.25 }}
            />
            {t}
            {t === part ? " (searched)" : ""}
          </span>
        ))}
        <span className="inline-flex items-center gap-1">
          <span className="inline-block size-2.5 rounded-full ring-2 ring-accent" /> this request
        </span>
      </div>
    </div>
  );
}
