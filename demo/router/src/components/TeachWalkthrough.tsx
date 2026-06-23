import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { Slider } from "@heroui/react";
import type { DemoData, DemoItem } from "../types";
import { SEED_COUNT } from "../demo-config";
import { qualClass } from "../format";
import { Button, Panel } from "./ui";
import { JsonBlock } from "./CodeJson";
import { StepGrid, StepWalkthrough } from "./StepWalkthrough";

const seedScores = (item: DemoItem): Record<string, number> =>
  Object.fromEntries(item.responses.map((r) => [r.model, r.score]));

const STEPS = [
  { id: "cold", title: "The store starts empty" },
  { id: "compare", title: "Send a prompt to every model" },
  { id: "score", title: "Score each answer" },
  { id: "rank", title: "Submit the scores" },
  { id: "warm", title: "Repeat until it's warm" },
  { id: "route", title: "Now it routes for you" },
] as const;

export function TeachWalkthrough({ demo, onSeeItRoute }: { demo: DemoData; onSeeItRoute: () => void }) {
  const item = demo.items[0];
  const [scores, setScores] = useState<Record<string, number>>(() => seedScores(item));
  // Records committed per task partition. The opening example (step 3) is the
  // first submission, filed under its own task; later submissions add to theirs.
  const [recordsByTask, setRecordsByTask] = useState<Record<string, number>>(() => ({
    [item.task]: item.responses.length,
  }));

  const tasks = useMemo(() => Array.from(new Set(demo.items.map((i) => i.task))), [demo]);
  const taskRecords = (t: string) => recordsByTask[t] ?? 0;
  const totalRecords = Object.values(recordsByTask).reduce((a, b) => a + b, 0);
  const taskWarm = (t: string) => taskRecords(t) >= SEED_COUNT;
  // Fill the least-populated partition next, so the three warm in turn on screen
  // instead of one filling first (the bundled prompts are grouped by task).
  const nextTask = tasks.reduce((a, b) => (taskRecords(a) <= taskRecords(b) ? a : b));
  const submitAnother = () => {
    const example = demo.items.find((it) => it.task === nextTask) ?? demo.items[0];
    setRecordsByTask((m) => ({ ...m, [nextTask]: (m[nextTask] ?? 0) + example.responses.length }));
  };

  const rankScores = useMemo(
    () => item.responses.map((r) => ({ ...r, score: scores[r.model] ?? r.score })),
    [item, scores],
  );

  return (
    <StepWalkthrough
      title="How you teach the router"
      description={
        <>
          A step-by-step walk through warming up the store from the gateway's preference API. Click{" "}
          <strong>Next</strong> (or use ← →).
        </>
      }
      steps={STEPS}
      footer={`${item.task} example`}
    >
      {(step) => (
        <>
          {step === 0 && (
            <StepGrid
              lede={
                <>
                  Otari learns <strong>per tenant</strong>, where a tenant is just the user behind the API key. You
                  teach and read the store through the gateway's <strong>preference API</strong>, and it is all scoped
                  to the key making the calls: preferences submitted under a key are stored for that key's user, and
                  chat requests served under the <em>same</em> key route against that user's records. That is what
                  "the same key you serve traffic with" means, train and serve with one key so the memory you build
                  is the memory that gets used. Each user keeps their own records, grouped by <code>task_id</code>; a
                  different key is a different tenant with separate pools. A brand-new tenant's store is empty, so
                  every request passes straight through until you teach it.
                </>
              }
            >
              <div className="flex flex-col gap-3">
                <Panel>
                  <span className="text-xs uppercase tracking-wide text-muted">How memory is organized</span>
                  <div className="mt-2 font-mono text-xs leading-relaxed">
                    <div>
                      API key <span className="text-muted">→</span> tenant{" "}
                      <span className="text-foreground">(one user)</span>
                    </div>
                    <div className="mt-1 pl-4 text-muted">its records, grouped by task_id:</div>
                    <div className="pl-6">
                      <span className="text-accent">support-bot</span> <span className="text-muted">→ own pool</span>
                    </div>
                    <div className="pl-6">
                      <span className="text-accent">code-review</span> <span className="text-muted">→ own pool</span>
                    </div>
                    <div className="pl-6">
                      <span className="opacity-60">(no task_id)</span> <span className="text-muted">→ shared pool</span>
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-muted">
                    A different key is a different tenant, with its own separate pools.
                  </p>
                </Panel>
                <Panel className="font-mono text-xs">
                  <div className="text-muted">GET /v1/router/status</div>
                  <JsonBlock
                    className="mt-2"
                    value={{
                      backend: "knn",
                      seed_count: SEED_COUNT,
                      default_pool: { records: 0, warm: false },
                      tasks: [],
                    }}
                  />
                  <p className="mt-3 font-sans text-muted">Empty store: nothing to route on yet.</p>
                </Panel>
              </div>
            </StepGrid>
          )}

          {step === 1 && (
            <StepGrid
              wide
              lede={
                <>
                  Pick a prompt and ask every candidate model for an answer in one call. The gateway fans it out so
                  you can compare real answers side by side. No scoring yet, just the raw responses.
                </>
              }
            >
              <div className="flex flex-col gap-3">
                <Panel className="font-mono text-xs">
                  <div className="text-muted">POST /v1/router/preferences/compare</div>
                  <JsonBlock className="mt-2" value={{ prompt: item.prompt, models: item.responses.map((r) => r.label) }} />
                </Panel>
                <ul className="grid gap-3 sm:grid-flow-col sm:auto-cols-fr">
                  {item.responses.map((r) => (
                    <li key={r.model} className="flex flex-col rounded-xl border border-border bg-surface p-3">
                      <span className="mb-2 truncate font-semibold">{r.label}</span>
                      <pre className="answer-text text-xs text-foreground/90">{r.text}</pre>
                    </li>
                  ))}
                </ul>
              </div>
            </StepGrid>
          )}

          {step === 2 && (
            <StepGrid
              wide
              lede={
                <>
                  Rate each answer from <strong>0</strong> (wrong) to <strong>1</strong> (great). In this demo a
                  gpt-5.4 judge pre-filled the scores; drag any slider to override. These ratings are the only quality
                  signal the router learns from, so they are yours to set (a human, or an automated judge).
                </>
              }
            >
              <Panel>
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-wide text-muted">Your scores</span>
                  <Button variant="ghost" size="sm" onPress={() => setScores(seedScores(item))}>
                    Reset to judge
                  </Button>
                </div>
                <ul className="mt-2 flex flex-col gap-4">
                  {item.responses.map((r) => {
                    const value = scores[r.model] ?? r.score;
                    return (
                      <li key={r.model}>
                        <div className="mb-1 flex items-center justify-between gap-2 text-sm">
                          <span className="min-w-0 flex-1 truncate font-semibold">{r.label}</span>
                          <span className={`tabular-nums font-semibold ${qualClass(value)}`}>{value.toFixed(2)}</span>
                        </div>
                        <pre className="answer-text mb-2 text-xs text-muted">{r.text}</pre>
                        <Slider
                          value={value}
                          onChange={(v) => setScores((s) => ({ ...s, [r.model]: v as number }))}
                          minValue={0}
                          maxValue={1}
                          step={0.05}
                          aria-label={`score for ${r.label}`}
                        >
                          <Slider.Track className="relative h-1.5 w-full rounded-full bg-surface-tertiary">
                            <Slider.Fill className="absolute h-full rounded-full bg-accent" />
                            <Slider.Thumb className="size-4 rounded-full bg-accent shadow" />
                          </Slider.Track>
                        </Slider>
                      </li>
                    );
                  })}
                </ul>
              </Panel>
            </StepGrid>
          )}

          {step === 3 && (
            <StepGrid
              lede={
                <>
                  Send the scores back. The gateway writes one <strong>routing-memory record</strong> per model: the
                  prompt's embedding paired with that model's score. That is the unit the kNN router later votes over.
                </>
              }
              aside={
                <Panel className="border-l-2 border-accent">
                  <p className="text-sm text-foreground/90">
                    <strong>Optional: split by use case.</strong> Memory is always scoped per tenant (the user behind
                    the API key), so two users never see each other's records, and that is usually all you need. Within
                    one tenant you can <em>optionally</em> add a <code>task_id</code> to keep unrelated use-cases apart,
                    so a support bot and a code reviewer never learn from each other:
                  </p>
                  <div className="mt-2 font-mono text-xs text-muted">
                    POST /v1/router/preferences/rank {"{"} ...,{" "}
                    <span className="text-accent">"task_id": "support-bot"</span> {"}"}
                  </div>
                  <p className="mt-2 text-sm text-foreground/90">
                    Then send <code>Otari-Router-Task: support-bot</code> on the requests you want routed against that
                    partition. Leave both off and everything shares one pool.
                  </p>
                </Panel>
              }
            >
              <div className="flex flex-col gap-3">
                <Panel className="font-mono text-xs">
                  <div className="text-muted">POST /v1/router/preferences/rank</div>
                  <JsonBlock
                    className="mt-2"
                    value={{
                      prompt: item.prompt,
                      scores: Object.fromEntries(rankScores.map((r) => [r.label, Number(r.score.toFixed(2))])),
                    }}
                  />
                  <div className="mt-3 text-muted">→ 200 OK</div>
                  <JsonBlock
                    className="mt-1"
                    value={{ recorded: rankScores.length, warm: totalRecords >= SEED_COUNT }}
                  />
                </Panel>
                <Panel>
                  <span className="text-xs uppercase tracking-wide text-muted">Records written to your pool</span>
                  <ul className="mt-2 flex flex-col gap-1 font-mono text-xs">
                    {rankScores.map((r) => (
                      <li key={r.model} className="flex items-center gap-2">
                        <span className="text-muted">record:</span>
                        <span>embed(prompt)</span>
                        <span className="text-muted">→</span>
                        <span>{r.label}</span>
                        <span className="text-muted">=</span>
                        <span className={qualClass(r.score)}>{r.score.toFixed(2)}</span>
                      </li>
                    ))}
                  </ul>
                </Panel>
              </div>
            </StepGrid>
          )}

          {step === 4 && (
            <StepGrid
              lede={
                <>
                  One example is not enough; repeat compare → score → rank for a handful of prompts. The{" "}
                  <strong>seed count</strong> applies to <strong>each pool on its own</strong>, so a pool stays in
                  pass-through until it alone crosses the threshold. If you used the optional <code>task_id</code>, each
                  task partition warms separately from the shared default pool; <code>/v1/router/status</code> reports
                  them all.
                </>
              }
            >
              <div className="flex flex-col gap-3">
                <Panel className="font-mono text-xs">
                  <div className="text-muted">GET /v1/router/status</div>
                  <JsonBlock
                    className="mt-2"
                    value={{
                      backend: "knn",
                      seed_count: SEED_COUNT,
                      default_pool: { records: totalRecords, warm: totalRecords >= SEED_COUNT },
                      tasks: tasks
                        .filter((t) => taskRecords(t) > 0)
                        .map((t) => ({ task_id: t, records: taskRecords(t), warm: taskWarm(t) })),
                    }}
                  />
                </Panel>
                <Panel>
                  <span className="text-xs uppercase tracking-wide text-muted">Per-partition warm-up</span>
                  <ul className="mt-2 flex flex-col gap-3">
                    {tasks.map((t) => (
                      <li key={t}>
                        <div className="mb-1 flex items-center justify-between text-xs">
                          <span className="font-mono">{t}</span>
                          <span className="tabular-nums text-muted">
                            {Math.min(taskRecords(t), SEED_COUNT)} / {SEED_COUNT}
                            {taskWarm(t) ? <span className="ml-2 text-success">✓ warm</span> : ""}
                          </span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-surface-tertiary">
                          <motion.div
                            className={`h-full rounded-full ${taskWarm(t) ? "bg-success" : "bg-accent"}`}
                            initial={false}
                            animate={{ width: `${Math.min(100, (taskRecords(t) / SEED_COUNT) * 100)}%` }}
                            transition={{ duration: 0.4, ease: "easeOut" }}
                          />
                        </div>
                      </li>
                    ))}
                  </ul>
                  <div className="mt-3 flex items-center justify-between gap-2">
                    <span className="text-xs text-muted">
                      Next example files into <span className="font-mono">{nextTask}</span>
                    </span>
                    <Button variant="secondary" size="sm" onPress={submitAnother}>
                      Submit another example →
                    </Button>
                  </div>
                </Panel>
                <p className="text-xs text-muted">
                  Each submission adds one record per scored model to that prompt's partition. The gateway gates on{" "}
                  <code>OTARI_ROUTER_SEED_COUNT</code> (default 20) per partition; a request with no task header routes
                  over the combined <code>default_pool</code>.
                </p>
              </div>
            </StepGrid>
          )}

          {step === 5 && (
            <StepGrid
              lede={
                <>
                  The store is warm. From now on a new prompt that looks like the ones you scored gets the cheapest
                  model that cleared your bar, automatically, on the normal <code>/v1/chat/completions</code> path.
                  Nothing changes in your app; you keep requesting your strong model. If you trained more than one
                  task, add the <code>Otari-Router-Task</code> header so the request routes against the matching
                  partition.
                </>
              }
            >
              <Panel>
                <div className="flex items-center justify-center gap-3 text-sm">
                  <span className="rounded-lg bg-surface-secondary px-3 py-2">you request the premium model</span>
                  <span className="text-muted">→</span>
                  <span className="rounded-lg bg-surface-secondary px-3 py-2">
                    Otari serves the <strong className="text-success">cheapest model good enough</strong>
                  </span>
                </div>
                <div className="mt-3 font-mono text-xs text-muted">
                  <span className="text-muted">Otari-Router-Task:</span> {item.task}{" "}
                  <span className="opacity-60"># route against the partition you trained</span>
                </div>
                <p className="mt-3 text-sm text-muted">
                  The response's <code>model</code> field reflects what actually ran, and usage and cost are
                  attributed to it. See exactly how the router picks:
                </p>
                <div className="mt-3 flex justify-center">
                  <Button variant="primary" onPress={onSeeItRoute}>
                    See it route →
                  </Button>
                </div>
              </Panel>
            </StepGrid>
          )}
        </>
      )}
    </StepWalkthrough>
  );
}
