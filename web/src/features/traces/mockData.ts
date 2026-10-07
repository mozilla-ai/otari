import type {
  Observation,
  ObservationLevel,
  ObservationType,
  Trace,
} from "./types"

// Deterministic so the demo looks the same on every load.
function rng(seed: number) {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const RATES: Record<
  string,
  { provider: string; inPer1k: number; outPer1k: number }
> = {
  "gpt-5": { provider: "openai", inPer1k: 0.00125, outPer1k: 0.01 },
  "gpt-4o-mini": { provider: "openai", inPer1k: 0.00015, outPer1k: 0.0006 },
  "claude-sonnet-4-5": {
    provider: "anthropic",
    inPer1k: 0.003,
    outPer1k: 0.015,
  },
  "gemini-2.5-pro": { provider: "google", inPer1k: 0.00125, outPer1k: 0.01 },
  "mistral-large": { provider: "mistral", inPer1k: 0.002, outPer1k: 0.006 },
}

function id(r: () => number, n = 8) {
  let s = ""
  while (s.length < n) s += Math.floor(r() * 16).toString(16)
  return s
}

interface Draft {
  key: string
  parent?: string
  type: ObservationType
  name: string
  start: number
  end: number
  level?: ObservationLevel
  statusMessage?: string
  model?: string
  tokens?: [number, number]
  input?: unknown
  output?: unknown
  metadata?: Record<string, string>
}

function build(
  traceId: string,
  drafts: Draft[],
  r: () => number,
): Observation[] {
  const ids = new Map(drafts.map((d) => [d.key, id(r)]))
  return drafts.map((d) => {
    const rate = d.model ? RATES[d.model] : undefined
    const [pt, ct] = d.tokens ?? [0, 0]
    return {
      id: ids.get(d.key) as string,
      traceId,
      parentObservationId: d.parent ? (ids.get(d.parent) ?? null) : null,
      type: d.type,
      name: d.name,
      startMs: d.start,
      endMs: d.end,
      level: d.level ?? "DEFAULT",
      statusMessage: d.statusMessage,
      model: d.model,
      provider: rate?.provider,
      promptTokens: d.tokens ? pt : undefined,
      completionTokens: d.tokens ? ct : undefined,
      inputCost:
        rate && d.level !== "ERROR"
          ? (pt / 1000) * rate.inPer1k
          : rate
            ? 0
            : undefined,
      outputCost:
        rate && d.level !== "ERROR"
          ? (ct / 1000) * rate.outPer1k
          : rate
            ? 0
            : undefined,
      input: d.input,
      output: d.output,
      metadata: d.metadata,
    }
  })
}

const NOW = Date.now()

// The deep trace the Log View is checked against: an agent that plans, searches,
// delegates to a sub-agent and answers. 12 observations, depth L0 to L3.
function chatbotTrace(): Trace {
  const r = rng(7)
  const traceId = "7f660c74a1b24e0d9c3b5f2e8d1a4c66"
  const question =
    "Why did our p95 latency double after the routing change, and which models are affected?"
  const drafts: Draft[] = [
    {
      key: "root",
      type: "AGENT",
      name: "handle-chatbot-message",
      start: 0,
      end: 37000,
      input: { message: question },
      output: {
        message:
          "Latency doubled because requests to gpt-5 now fall back to claude-sonnet-4-5 on 429s...",
      },
    },
    {
      key: "guard",
      parent: "root",
      type: "SPAN",
      name: "check-input-policy",
      start: 20,
      end: 410,
      metadata: { policy: "pii-redact", verdict: "pass" },
    },
    {
      key: "chain",
      parent: "root",
      type: "SPAN",
      name: "plan-and-answer",
      start: 420,
      end: 36600,
    },
    {
      key: "plan",
      parent: "chain",
      type: "GENERATION",
      name: "chat",
      start: 430,
      end: 3100,
      model: "gpt-5",
      tokens: [1180, 342],
      input: [
        { role: "system", content: "You are an SRE assistant." },
        { role: "user", content: question },
      ],
      output: {
        role: "assistant",
        content:
          "Plan: 1) search recent incidents 2) pull routing config 3) compare latency by model.",
      },
    },
    {
      key: "search",
      parent: "chain",
      type: "TOOL",
      name: "web_search",
      start: 3200,
      end: 6800,
      input: { query: "gateway fallback latency 429 retry" },
      output: { results: 6 },
      metadata: { server: "web-search" },
    },
    {
      key: "vec",
      parent: "chain",
      type: "SPAN",
      name: "runbook-lookup",
      start: 6900,
      end: 7300,
      input: { query: "routing fallback policy" },
      output: { documents: 4 },
    },
    {
      key: "agent",
      parent: "chain",
      type: "AGENT",
      name: "research-subagent",
      start: 7400,
      end: 30200,
    },
    {
      key: "g1",
      parent: "agent",
      type: "GENERATION",
      name: "chat",
      start: 7450,
      end: 12200,
      model: "claude-sonnet-4-5",
      tokens: [3920, 611],
      input: [
        { role: "user", content: "List the routing changes since Monday." },
      ],
      output: {
        role: "assistant",
        tool_calls: [{ name: "mcp:github.search_issues" }],
      },
    },
    {
      key: "t1",
      parent: "agent",
      type: "TOOL",
      name: "mcp:github.search_issues",
      start: 12300,
      end: 15900,
      input: { repo: "acme/gateway", q: "routing fallback" },
      output: { total: 3 },
      metadata: { server: "github" },
    },
    {
      key: "g2",
      parent: "agent",
      type: "GENERATION",
      name: "chat",
      start: 16000,
      end: 30100,
      model: "claude-sonnet-4-5",
      tokens: [6410, 1822],
      input: [{ role: "tool", content: "3 issues" }],
      output: {
        role: "assistant",
        content: "Fallback to claude-sonnet-4-5 was enabled on Tuesday.",
      },
    },
    {
      key: "final",
      parent: "chain",
      type: "GENERATION",
      name: "chat",
      start: 30300,
      end: 36500,
      model: "gpt-5",
      tokens: [8120, 904],
      input: [{ role: "user", content: question }],
      output: {
        role: "assistant",
        content:
          "Latency doubled because requests to gpt-5 now fall back to claude-sonnet-4-5 on 429s.",
      },
    },
    {
      key: "sent",
      parent: "root",
      type: "EVENT",
      name: "response-sent",
      start: 36900,
      end: 36900,
    },
  ]
  return {
    id: traceId,
    name: "handle-chatbot-message",
    timestamp: new Date(NOW - 3 * 60_000).toISOString(),
    userId: "dana@acme.io",
    sessionId: "sess-4f21",
    environment: "production",
    tags: ["chatbot", "sre"],
    input: { message: question },
    output: {
      message:
        "Latency doubled because requests to gpt-5 now fall back to claude-sonnet-4-5 on 429s.",
    },
    metadata: { release: "2026.10.3", region: "eu-west-1" },
    observations: build(traceId, drafts, r),
  }
}

const NAMES = [
  "chat-completion",
  "support-agent",
  "summarize-document",
  "sql-assistant",
  "code-review-bot",
  "extract-invoice",
  "triage-ticket",
  "translate-batch",
]
const USERS = [
  "dana@acme.io",
  "marcus@acme.io",
  "priya@acme.io",
  "ci-bot",
  "svc-support",
]
const TAGS = ["production", "support", "batch", "eval", "internal"]
const MODEL_LIST = Object.keys(RATES)

function generated(i: number): Trace {
  const r = rng(1000 + i)
  const traceId = `${id(r, 32)}`
  const name = NAMES[Math.floor(r() * NAMES.length)]
  const model = MODEL_LIST[Math.floor(r() * MODEL_LIST.length)]
  const t0 = NOW - (i + 1) * (4 + Math.floor(r() * 25)) * 60_000
  const drafts: Draft[] = []
  const kind =
    i % 9 === 4
      ? "error"
      : i % 7 === 2 || i % 11 === 5
        ? "fallback"
        : r() < 0.35
          ? "agent"
          : "simple"
  const p = 400 + Math.floor(r() * 3000)
  const c = 80 + Math.floor(r() * 900)
  const prompt = `Request ${i + 1}: ${name.replace(/-/g, " ")}`
  if (kind === "simple") {
    const d = 600 + Math.floor(r() * 4200)
    drafts.push({
      key: "g",
      type: "GENERATION",
      name: "chat",
      start: 0,
      end: d,
      model,
      tokens: [p, c],
      input: [{ role: "user", content: prompt }],
      output: { role: "assistant", content: "Done." },
    })
  } else if (kind === "fallback") {
    const alt = model === "claude-sonnet-4-5" ? "gpt-5" : "claude-sonnet-4-5"
    drafts.push(
      { key: "root", type: "SPAN", name: name, start: 0, end: 7800 },
      {
        key: "route",
        parent: "root",
        type: "SPAN",
        name: "routing",
        start: 10,
        end: 7700,
        metadata: { policy: "fallback-chain", attempts: "2" },
      },
      {
        key: "a1",
        parent: "route",
        type: "GENERATION",
        name: "chat (attempt 1)",
        start: 20,
        end: 2100,
        level: "WARNING",
        statusMessage: "429 Too Many Requests, retried on the fallback",
        model,
        tokens: [p, 0],
        input: [{ role: "user", content: prompt }],
      },
      {
        key: "a2",
        parent: "route",
        type: "GENERATION",
        name: "chat (attempt 2)",
        start: 2200,
        end: 7600,
        level: "WARNING",
        statusMessage: "Served by fallback provider",
        model: alt,
        tokens: [p, c],
        input: [{ role: "user", content: prompt }],
        output: { role: "assistant", content: "Done." },
      },
    )
  } else if (kind === "error") {
    drafts.push(
      {
        key: "root",
        type: "SPAN",
        name: name,
        start: 0,
        end: 3400,
        level: "ERROR",
        statusMessage: "tool call failed",
      },
      {
        key: "g",
        parent: "root",
        type: "GENERATION",
        name: "chat",
        start: 10,
        end: 2200,
        model,
        tokens: [p, c],
        input: [{ role: "user", content: prompt }],
        output: { role: "assistant", tool_calls: [{ name: "lookup_order" }] },
      },
      {
        key: "t",
        parent: "root",
        type: "TOOL",
        name: "mcp:orders.lookup_order",
        start: 2300,
        end: 3400,
        level: "ERROR",
        statusMessage: "MCP server timed out after 1000ms",
        input: { order_id: "A-10293" },
      },
    )
  } else {
    const n = 1 + Math.floor(r() * 3)
    let cur = 40
    drafts.push({ key: "root", type: "AGENT", name, start: 0, end: 0 })
    for (let k = 0; k < n; k++) {
      const gd = 700 + Math.floor(r() * 3000)
      drafts.push({
        key: `g${k}`,
        parent: "root",
        type: "GENERATION",
        name: "chat",
        start: cur,
        end: cur + gd,
        model,
        tokens: [p + k * 300, c],
        input: [{ role: "user", content: prompt }],
        output: { role: "assistant", content: "Step done." },
      })
      cur += gd + 20
      const td = 300 + Math.floor(r() * 2200)
      drafts.push({
        key: `t${k}`,
        parent: "root",
        type: "TOOL",
        name: k % 2 ? "web_search" : "mcp:docs.search",
        start: cur,
        end: cur + td,
        input: { query: "..." },
        output: { hits: 3 },
      })
      cur += td + 20
    }
    drafts[0].end = cur
  }
  return {
    id: traceId,
    name,
    timestamp: new Date(t0).toISOString(),
    userId: USERS[Math.floor(r() * USERS.length)],
    sessionId: `sess-${id(r, 4)}`,
    environment: r() < 0.85 ? "production" : "staging",
    tags: [TAGS[Math.floor(r() * TAGS.length)]],
    input: { message: prompt },
    output: kind === "error" ? null : { message: "Done." },
    metadata: { release: "2026.10.3" },
    observations: build(traceId, drafts, r),
  }
}

export const MOCK_TRACES: Trace[] = [
  chatbotTrace(),
  ...Array.from({ length: 41 }, (_, i) => generated(i)),
].sort((a, b) => b.timestamp.localeCompare(a.timestamp))
