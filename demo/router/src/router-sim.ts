// Client-side reproduction of the gateway's kNN routing decision, used by the
// Walkthrough to demonstrate the router on the bundled prompts with no backend.
// It mirrors gateway/services/knn_router.py:
//   score(m) = mean_quality(m | neighbors) - alpha * normalized_cost(m)
// picking the highest-scoring candidate, with the requested model kept as the
// guaranteed fallback. This is leave-one-out over the bundled set: each prompt is
// routed using the OTHER prompts as the routing memory, so it is an honest (if
// tiny) run of the real algorithm, not a canned result.
import type { DemoData, DemoItem } from "./types";

function unit(v: number[]): number[] {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export interface NeighborInfo {
  item: DemoItem;
  sim: number;
  quality: Record<string, number>; // model id -> judge quality score on this neighbor
}

export interface CandidateScore {
  model: string;
  meanQuality: number; // mean over the k neighbors
  price: number;
  normCost: number; // price normalized to [0,1] across the pool
  score: number; // meanQuality - alpha * normCost
  winner: boolean;
}

export interface RouteExplanation {
  item: DemoItem;
  requested: string; // the strong default the caller would ask for
  winner: string; // model the router picked
  neighbors: NeighborInfo[];
  candidates: CandidateScore[];
  servedScore: number; // selected prompt's judge quality score for the winner
  servedAnswer: string; // winner's answer text for the selected prompt
}

/** Route a single bundled prompt (leave-one-out) and return the full kNN breakdown. */
export function explainOne(demo: DemoData, index: number, alpha: number, k: number): RouteExplanation {
  const pool = demo.models.map((m) => m.id);
  const price = Object.fromEntries(demo.models.map((m) => [m.id, m.price]));
  const prices = pool.map((m) => price[m]);
  const lo = Math.min(...prices);
  const hi = Math.max(...prices);
  const span = hi - lo;
  const normCost = (m: string) => (span === 0 ? 0 : (price[m] - lo) / span);
  const strongest = pool.reduce((a, b) => (price[a] >= price[b] ? a : b));

  const vecs = demo.items.map((it) => unit(it.embedding));
  const scoreOf = (it: DemoItem, model: string) => it.responses.find((r) => r.model === model)?.score ?? 0;

  const item = demo.items[index];
  const sims = demo.items.map((_, j) => ({ j, s: j === index ? -Infinity : dot(vecs[index], vecs[j]) }));
  sims.sort((a, b) => b.s - a.s);
  const top = sims.slice(0, k);
  const neighborItems = top.map((x) => demo.items[x.j]);

  const neighbors: NeighborInfo[] = top.map((x) => ({
    item: demo.items[x.j],
    sim: x.s,
    quality: Object.fromEntries(pool.map((m) => [m, scoreOf(demo.items[x.j], m)])),
  }));

  let winner = strongest;
  let best = -Infinity;
  const candidates: CandidateScore[] = pool.map((m) => {
    const qs = neighborItems.map((nb) => scoreOf(nb, m));
    const meanQuality = qs.reduce((a, b) => a + b, 0) / (qs.length || 1);
    const score = meanQuality - alpha * normCost(m);
    if (score > best) {
      best = score;
      winner = m;
    }
    return { model: m, meanQuality, price: price[m], normCost: normCost(m), score, winner: false };
  });
  for (const c of candidates) c.winner = c.model === winner;

  return {
    item,
    requested: strongest,
    winner,
    neighbors,
    candidates,
    servedScore: scoreOf(item, winner),
    servedAnswer: item.responses.find((r) => r.model === winner)?.text ?? "",
  };
}
