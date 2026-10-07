import { pickText } from "./mockText"
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
  id?: string
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
  const ids = new Map(drafts.map((d) => [d.key, d.id ?? id(r)]))
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
  const answer =
    "Latency doubled because requests to gpt-5 now fall back to claude-sonnet-4-5 on 429s."
  const chat = (
    key: string,
    parent: string,
    start: number,
    end: number,
    model: string,
    tokens: [number, number],
    content: string,
  ): Draft => ({
    key,
    parent,
    type: "GENERATION",
    name: "chat",
    start,
    end,
    model,
    tokens,
    input: [{ role: "user", content: question }],
    output: { role: "assistant", content },
  })
  const drafts: Draft[] = [
    {
      key: "root",
      id: traceId,
      type: "AGENT",
      name: "handle-chatbot-message",
      start: 0,
      end: 37000,
      input: { message: question },
      output: { message: answer },
    },
    {
      key: "policy",
      parent: "root",
      type: "SPAN",
      name: "get-routing-policy",
      start: 0,
      end: 293,
    },
    {
      key: "mcp",
      parent: "root",
      type: "SPAN",
      name: "create-mcp-client",
      start: 0,
      end: 77,
    },
    {
      key: "qa",
      parent: "root",
      type: "AGENT",
      name: "qa-chatbot",
      start: 0,
      end: 36500,
    },
    {
      key: "s1",
      parent: "qa",
      type: "SPAN",
      name: "step",
      start: 0,
      end: 3000,
    },
    chat(
      "c1",
      "s1",
      0,
      2900,
      "gpt-5",
      [1180, 342],
      "Plan: compare latency by model, then check the routing config.",
    ),
    {
      key: "t1",
      parent: "s1",
      type: "TOOL",
      name: "getGatewayOverview",
      start: 3000,
      end: 3116,
      input: { window: "7d" },
      output: { p95_ms: 4210, fallback_rate: 0.31 },
      metadata: { server: "otari-mcp" },
    },
    {
      key: "s2",
      parent: "qa",
      type: "SPAN",
      name: "step",
      start: 3000,
      end: 27900,
    },
    chat(
      "c2",
      "s2",
      3000,
      5600,
      "claude-sonnet-4-5",
      [3920, 611],
      "Searching the docs for the fallback policy.",
    ),
    {
      key: "t2",
      parent: "s2",
      type: "TOOL",
      name: "searchGatewayDocs",
      start: 6000,
      end: 28300,
      input: { query: "routing fallback policy 429 retry" },
      output: { results: 6 },
      metadata: { server: "otari-mcp" },
    },
    {
      key: "s3",
      parent: "qa",
      type: "SPAN",
      name: "step",
      start: 28000,
      end: 36600,
    },
    chat("c3", "s3", 28000, 36600, "gpt-5", [8120, 904], answer),
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
  const text = pickText(name, i)
  const prompt = text.input
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
      output: { role: "assistant", content: text.output },
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
        output: { role: "assistant", content: text.output },
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
        output: { role: "assistant", content: text.output },
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
        input: { query: text.input.slice(0, 48) },
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
    output: kind === "error" ? null : { message: text.output },
    metadata: { release: "2026.10.3" },
    observations: build(traceId, drafts, r),
  }
}

export const MOCK_TRACES: Trace[] = [
  chatbotTrace(),
  ...Array.from({ length: 41 }, (_, i) => generated(i)),
].sort((a, b) => b.timestamp.localeCompare(a.timestamp))
