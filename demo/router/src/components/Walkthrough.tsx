import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { Chip, Slider } from "@heroui/react";
import { explainOne } from "../router-sim";
import type { DemoData } from "../types";
import { SEED_COUNT } from "../demo-config";
import { qualClass } from "../format";
import { Panel } from "./ui";
import { KnnGraph } from "./KnnGraph";
import { JsonBlock } from "./CodeJson";
import { StepGrid, StepWalkthrough } from "./StepWalkthrough";

const labelOf = (demo: DemoData, id: string) => demo.models.find((m) => m.id === id)?.label ?? id;
const spreadOf = (r: { score: number }[]) => Math.max(...r.map((x) => x.score)) - Math.min(...r.map((x) => x.score));

// Curated entry point: the hard prompt with the widest model spread (it routes
// to a cheaper but still capable model). Computed from the data so it survives a
// regeneration.
function curatedHardIndex(demo: DemoData): number {
  let hard = 0;
  let hardSpread = -1;
  demo.items.forEach((it, i) => {
    const s = spreadOf(it.responses);
    if (it.task.startsWith("hard") && s > hardSpread) {
      hardSpread = s;
      hard = i;
    }
  });
  return hard;
}

const STEPS = [
  { id: "request", title: "The request arrives" },
  { id: "gates", title: "Is it safe to route?" },
  { id: "embed", title: "Embed the prompt" },
  { id: "knn", title: "Find similar past prompts" },
  { id: "quality", title: "Predict quality per model" },
  { id: "cost", title: "Weigh quality against cost" },
  { id: "decision", title: "Pick the cheapest capable model" },
  { id: "result", title: "Result" },
] as const;

const K = 5;

export function Walkthrough({ demo }: { demo: DemoData }) {
  const index = useMemo(() => curatedHardIndex(demo), [demo]);
  const [alpha, setAlpha] = useState(0.3);
  const exp = useMemo(() => explainOne(demo, index, alpha, K), [demo, index, alpha]);

  const price = Object.fromEntries(demo.models.map((m) => [m.id, m.price]));
  const requested = exp.requested;
  const winner = exp.winner;
  const cheapest = demo.models.reduce((a, b) => (a.price <= b.price ? a : b)).id;
  const rerouted = winner !== requested;
  const savings = price[requested] ? Math.round(((price[requested] - price[winner]) / price[requested]) * 100) : 0;
  const dominantTask = mode(exp.neighbors.map((nb) => nb.item.task));
  const weakest = [...exp.candidates].sort((a, b) => a.meanQuality - b.meanQuality)[0];

  return (
    <StepWalkthrough
      title="How Otari routes a request"
      description={
        <>
          A step-by-step walk through the kNN router's decision. Click <strong>Next</strong> (or use ← →).
        </>
      }
      steps={STEPS}
      footer="Hard prompt example"
    >
      {(step) => (
        <>
          {step === 0 && (
            <StepGrid
              lede={
                <>
                  A caller hits <code>/v1/chat/completions</code> asking for{" "}
                  <strong>{labelOf(demo, requested)}</strong>, the safe premium default. Whether the gateway routes is
                  a server setting (<code>OTARI_ROUTER_BACKEND=knn</code> turns it on for every request), but a caller
                  can override it per request with the <code>Otari-Router</code> header: <code>off</code> pins this one
                  call to the requested model, <code>on</code> uses the server default. The optional{" "}
                  <code>Otari-Router-Task</code> header picks which slice of memory to consult, here{" "}
                  <strong>{exp.item.task}</strong>, so unrelated use-cases stay separate (more on that in a moment).
                  With routing active, let's watch it decide whether a cheaper model would do.
                </>
              }
            >
              <Panel className="font-mono text-xs">
                <div className="text-muted">POST /v1/chat/completions</div>
                <div>
                  <span className="text-muted">Otari-Router:</span> on{" "}
                  <span className="opacity-60"># optional; omit to use the server default</span>
                </div>
                <div>
                  <span className="text-muted">Otari-Router-Task:</span> {exp.item.task}{" "}
                  <span className="opacity-60"># optional; omit to search the whole pool</span>
                </div>
                <JsonBlock
                  className="mt-2"
                  value={{ model: labelOf(demo, requested), messages: [{ role: "user", content: exp.item.prompt }] }}
                />
              </Panel>
            </StepGrid>
          )}

          {step === 1 && (
            <StepGrid
              lede={
                <>
                  Before touching a model, the gateway runs a few guards. If any fails it passes the request
                  straight through to <strong>{labelOf(demo, requested)}</strong>, so routing is never worse than
                  leaving it off.
                </>
              }
            >
              <Panel>
                <ul className="flex flex-col gap-2 text-sm">
                  <Gate ok label="Router backend enabled" detail="OTARI_ROUTER_BACKEND = knn" />
                  <Gate ok label="No gateway-managed tools on the request" detail="tool calls always pass through" />
                  <Gate
                    ok
                    label="Task partition warm"
                    detail={`the ${exp.item.task} partition has ≥ ${SEED_COUNT} records`}
                  />
                  <Gate ok label="Prompt has embeddable text" detail="a task signal to match on" />
                </ul>
                <p className="mt-3 text-xs text-success">All clear. Routing proceeds.</p>
              </Panel>
            </StepGrid>
          )}

          {step === 2 && (
            <StepGrid
              lede={
                <>
                  The latest user message is embedded with <code>text-embedding-3-small</code> into a 256-dimension
                  vector. That vector is the query point in this tenant's <strong>{exp.item.task}</strong> partition.
                </>
              }
            >
              <Panel>
                <p className="mb-3 whitespace-pre-wrap rounded-lg bg-surface-secondary p-2 text-sm">{exp.item.prompt}</p>
                <div className="text-xs text-muted">embed →</div>
                <Sparkline values={exp.item.embedding.slice(0, 48)} />
                <div className="mt-1 text-right text-xs text-muted">256-d vector (first 48 dims shown)</div>
              </Panel>
            </StepGrid>
          )}

          {step === 3 && (
            <StepGrid
              wide
              lede={
                <>
                  A cosine k-nearest-neighbors search finds the <strong>{K}</strong> most similar prompts the store
                  has already seen scored. Because the request named the <strong>{exp.item.task}</strong> partition,
                  only that partition is searched, another user's or another use-case's vectors are never candidates,
                  so every neighbor below is <strong>{dominantTask}</strong> work. The router then assumes the new
                  prompt behaves like those neighbors.
                </>
              }
            >
              <div className="grid gap-4 lg:grid-cols-2">
                <KnnGraph demo={demo} exp={exp} k={K} />
                <ol className="flex flex-col gap-2">
                  {exp.neighbors.map((nb) => (
                    <li key={nb.item.id} className="rounded-xl border border-border p-2">
                      <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
                        <Chip size="sm">sim {nb.sim.toFixed(2)}</Chip>
                        <Chip size="sm">{nb.item.task}</Chip>
                        <span className="flex flex-wrap gap-x-3 tabular-nums text-muted">
                          {demo.models.map((m) => (
                            <span key={m.id}>
                              {m.label}: <strong className={qualClass(nb.quality[m.id])}>{nb.quality[m.id].toFixed(2)}</strong>
                            </span>
                          ))}
                        </span>
                      </div>
                      <p className="line-clamp-2 whitespace-pre-wrap text-sm text-muted">{nb.item.prompt}</p>
                    </li>
                  ))}
                </ol>
              </div>
            </StepGrid>
          )}

          {step === 4 && (
            <StepGrid
              lede={
                <>
                  For each candidate, average its quality across those neighbors. That is the router's prediction of
                  how each model handles <em>this kind</em> of prompt, no need to actually call them.{" "}
                  <strong>{labelOf(demo, weakest.model)}</strong> clearly lags here.
                </>
              }
            >
              <Panel>
                <div className="mb-2 text-xs uppercase tracking-wide text-muted">Predicted quality (mean over neighbors)</div>
                <div className="flex flex-col gap-2">
                  {exp.candidates.map((c) => (
                    <Bar key={c.model} label={labelOf(demo, c.model)} value={c.meanQuality} tone={qualBar(c.meanQuality)} />
                  ))}
                </div>
              </Panel>
            </StepGrid>
          )}

          {step === 5 && (
            <StepGrid
              lede={
                <>
                  Now fold in price. Each model scores{" "}
                  <code className="whitespace-nowrap">quality − α · cost</code>. The dial <strong>α</strong> is the
                  cost-vs-quality knob: <strong>0</strong> ignores price (always the best answer), higher leans
                  cheaper. Drag it and watch the winner move.
                </>
              }
            >
              <Panel>
                <div className="mb-3">
                  <div className="mb-1 flex items-center justify-between text-xs text-muted">
                    <span>Cost dial (α)</span>
                    <span className="font-semibold tabular-nums text-foreground">{alpha.toFixed(2)}</span>
                  </div>
                  <Slider value={alpha} onChange={(v) => setAlpha(v as number)} minValue={0} maxValue={1.5} step={0.05} aria-label="alpha">
                    <Slider.Track className="relative h-1.5 w-full rounded-full bg-surface-tertiary">
                      <Slider.Fill className="absolute h-full rounded-full bg-accent" />
                      <Slider.Thumb className="size-4 rounded-full bg-accent shadow" />
                    </Slider.Track>
                  </Slider>
                </div>
                <div className="flex flex-col gap-2">
                  {exp.candidates.map((c) => (
                    <div key={c.model} className={`rounded-lg px-2 py-1.5 ${c.winner ? "ring-1 ring-accent" : ""}`}>
                      <div className="flex items-center justify-between text-sm">
                        <span className={c.winner ? "font-semibold" : ""}>
                          {labelOf(demo, c.model)}
                          {c.winner ? " ★" : ""}
                        </span>
                        <span className="flex gap-3 text-xs tabular-nums text-muted">
                          <span>q {c.meanQuality.toFixed(2)}</span>
                          <span>cost {c.normCost.toFixed(2)}</span>
                          <span className={c.winner ? "font-semibold text-accent" : "text-foreground"}>
                            = {c.score.toFixed(2)}
                          </span>
                        </span>
                      </div>
                      <Bar value={Math.max(0, c.score)} tone={c.winner ? "accent" : "muted"} thin />
                    </div>
                  ))}
                </div>
              </Panel>
            </StepGrid>
          )}

          {step === 6 && (
            <StepGrid
              lede={
                <>
                  Highest score wins. Here it serves{" "}
                  <strong className={rerouted ? "text-success" : ""}>{labelOf(demo, winner)}</strong>: about as good
                  as <strong>{labelOf(demo, requested)}</strong> on this region but far cheaper, while{" "}
                  <strong>{labelOf(demo, cheapest)}</strong> was too weak to trust. A confidence floor (off by
                  default) can override the pick and lead with <strong>{labelOf(demo, requested)}</strong> when
                  neighbors disagree.
                </>
              }
            >
              <Panel>
                <div className="mb-2 text-xs uppercase tracking-wide text-muted">Models ranked by score</div>
                <ol className="flex flex-col gap-2">
                  {[...exp.candidates]
                    .sort((a, b) => b.score - a.score)
                    .map((c, i) => (
                      <li
                        key={c.model}
                        className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm ${
                          i === 0 ? "border-accent ring-1 ring-accent" : "border-border"
                        }`}
                      >
                        <span className="w-5 text-muted">{i + 1}.</span>
                        <span className={i === 0 ? "font-semibold" : ""}>{labelOf(demo, c.model)}</span>
                        <span className="ml-auto tabular-nums text-xs text-muted">score {c.score.toFixed(2)}</span>
                        {i === 0 && <Chip size="sm">served</Chip>}
                      </li>
                    ))}
                </ol>
              </Panel>
            </StepGrid>
          )}

          {step === 7 && (
            <StepGrid
              lede={
                <>
                  The caller asked for <strong>{labelOf(demo, requested)}</strong> and got a correct answer from{" "}
                  <strong>{labelOf(demo, winner)}</strong>
                  {rerouted ? (
                    <>
                      {" "}at <strong className="text-success">{savings}% lower cost</strong>, with no quality loss
                      on this prompt.
                    </>
                  ) : (
                    <> — the router kept the premium model because nothing cheaper was good enough here.</>
                  )}{" "}
                  Multiply that across real traffic and the easy/cheap regions fund the few prompts that truly need
                  the premium model.
                </>
              }
            >
              <Panel className={rerouted ? "ring-1 ring-success/40" : ""}>
                <div className="flex items-center justify-center gap-3 text-sm">
                  <span className="rounded-lg bg-surface-secondary px-3 py-2">
                    requested <strong>{labelOf(demo, requested)}</strong>
                  </span>
                  <span className="text-muted">→</span>
                  <span className="rounded-lg bg-surface-secondary px-3 py-2">
                    served <strong className={rerouted ? "text-success" : ""}>{labelOf(demo, winner)}</strong>
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-center">
                  <div className="rounded-lg border border-border p-3">
                    <div className="text-2xl font-semibold text-success">{rerouted ? `${savings}%` : "0%"}</div>
                    <div className="text-xs text-muted">cheaper than the requested model</div>
                  </div>
                  <div className="rounded-lg border border-border p-3">
                    <div className={`text-2xl font-semibold ${qualClass(exp.servedScore)}`}>{exp.servedScore.toFixed(2)}</div>
                    <div className="text-xs text-muted">judged quality of the served answer</div>
                  </div>
                </div>
              </Panel>
            </StepGrid>
          )}
        </>
      )}
    </StepWalkthrough>
  );
}

// ---- small presentational helpers ------------------------------------------

function Gate({ ok, label, detail }: { ok: boolean; label: string; detail: string }) {
  return (
    <li className="flex items-start gap-2">
      <span className={`mt-0.5 grid size-4 place-items-center rounded-full text-[10px] ${ok ? "bg-success text-white" : "bg-danger text-white"}`}>
        {ok ? "✓" : "✕"}
      </span>
      <span>
        {label} <span className="text-muted">— {detail}</span>
      </span>
    </li>
  );
}

function Bar({ label, value, tone, thin }: { label?: string; value: number; tone: "good" | "warn" | "bad" | "accent" | "muted"; thin?: boolean }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const bg = {
    good: "bg-success",
    warn: "bg-warning",
    bad: "bg-danger",
    accent: "bg-accent",
    muted: "bg-surface-tertiary",
  }[tone];
  return (
    <div>
      {label && (
        <div className="mb-1 flex items-center justify-between text-sm">
          <span>{label}</span>
          <span className="tabular-nums text-muted">{value.toFixed(2)}</span>
        </div>
      )}
      <div className={`overflow-hidden rounded-full bg-surface-tertiary ${thin ? "h-1.5" : "h-2.5"}`}>
        <motion.div
          className={`h-full rounded-full ${bg}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  const max = Math.max(...values.map((v) => Math.abs(v))) || 1;
  return (
    <div className="flex h-16 items-center gap-[2px]">
      {values.map((v, i) => {
        const h = Math.max(2, (Math.abs(v) / max) * 28);
        return (
          <div key={i} className="flex flex-1 flex-col justify-center" style={{ height: 56 }}>
            <motion.div
              className={`w-full rounded-sm ${v >= 0 ? "bg-accent" : "bg-accent/50"}`}
              initial={{ height: 0 }}
              animate={{ height: h }}
              transition={{ duration: 0.3, delay: i * 0.006 }}
              style={{ alignSelf: v >= 0 ? "flex-end" : "flex-start" }}
            />
          </div>
        );
      })}
    </div>
  );
}

function qualBar(v: number): "good" | "warn" | "bad" {
  return v >= 0.85 ? "good" : v >= 0.5 ? "warn" : "bad";
}

function mode(xs: string[]): string {
  const counts = new Map<string, number>();
  for (const x of xs) counts.set(x, (counts.get(x) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "";
}
