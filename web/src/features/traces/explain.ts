import type { TraceSpan } from "@/client"

import { formatDuration } from "./format"
import type { GuardrailVerdict } from "./gateway"
import { clip, clipPath } from "./traceModel"

// What a span means, in a sentence and a few facts, for the panel a picked span
// opens. Built from what the span already holds; nothing here is stored.
export interface Explanation {
  sentence: string
  facts: [term: string, value: string][]
}

const DECIDED_BY: Record<string, string> = {
  config: "a permission rule in your settings",
  hook: "a hook",
  user: "the user",
}

// Claude Code reports a user's decision as user_reject, user_temporary and so on.
function decisionSource(span: TraceSpan): unknown {
  const source = span.attributes["otari.tool.decision_source"]
  return typeof source === "string" && source.startsWith("user_")
    ? "user"
    : source
}

const STOPPED_BECAUSE: Record<string, string> = {
  tool_use: "It asked to use a tool",
  end_turn: "It finished its answer",
  max_tokens: "It reached its output limit",
  stop_sequence: "It reached a stop sequence",
  pause_turn: "It paused a long-running turn",
  refusal: "It refused",
}

function took(span: TraceSpan): string | null {
  return span.duration_ms === null ? null : formatDuration(span.duration_ms)
}

function count(value: unknown): string | null {
  return typeof value === "number" ? value.toLocaleString("en-US") : null
}

function facts(...pairs: [string, string | null | undefined][]) {
  return pairs.filter((pair): pair is [string, string] => Boolean(pair[1]))
}

// What a tool call was asked to do, the way its row names it.
function asked(tool: TraceSpan | undefined): string {
  if (!tool) return "the tool"
  const name = tool.tool_name ?? "the tool"
  if (tool.content.command) return `${name} on ${clip(tool.content.command)}`
  if (tool.content.file_path)
    return `${name} on ${clipPath(tool.content.file_path)}`
  return name
}

function permission(span: TraceSpan, tool: TraceSpan | undefined): Explanation {
  const decision = span.attributes["otari.tool.decision"]
  const source = decisionSource(span)
  const by =
    typeof source === "string" ? (DECIDED_BY[source] ?? source) : "something"
  const after = took(span)
  if (decision !== "accept" && decision !== "reject") {
    return {
      sentence: `Claude Code checked whether it could run ${asked(tool)}. It did not say what was decided.`,
      facts: facts(["Took", after]),
    }
  }
  const allowed = decision === "accept"
  const sentence =
    source === "user"
      ? `Claude Code asked the user whether to run ${asked(tool)}, and they ${allowed ? "allowed" : "refused"} it${after ? ` after ${after}` : ""}.`
      : `${allowed ? "Allowed" : "Refused"} by ${by}${after ? ` in ${after}` : ""}: ${asked(tool)}.`
  return {
    sentence,
    facts: facts(
      ["Decision", allowed ? "Allowed" : "Refused"],
      ["Decided by", typeof source === "string" ? by : null],
      [source === "user" ? "The user took" : "Took", after],
    ),
  }
}

// A returned error is a result the agent read, so the call still ran rather than failed.
function returnedError(span: TraceSpan): boolean {
  return span.attributes["otari.tool.succeeded"] === false
}

function ran(span: TraceSpan): Explanation {
  const after = took(span)
  const result =
    span.outcome === "error"
      ? "failed"
      : returnedError(span)
        ? "returned an error"
        : "succeeded"
  return {
    sentence: `Ran${after ? ` for ${after}` : ""} and ${result}.`,
    facts: [],
  }
}

function tool(
  span: TraceSpan,
  blockedBy: readonly string[],
  gates: GateTally | undefined,
): Explanation {
  const source = decisionSource(span)
  const decision = span.attributes["otari.tool.decision"]
  const decided =
    blockedBy.length > 0
      ? `Blocked by ${blockedBy.join(", ")}`
      : typeof source === "string"
        ? `${decision === "reject" ? "Refused" : "Allowed"} by ${DECIDED_BY[source] ?? source}`
        : null
  const outcome =
    span.outcome === "blocked"
      ? "It never ran."
      : span.outcome === "error"
        ? "It ran and failed."
        : returnedError(span)
          ? "It ran and returned an error, which the agent read."
          : "It ran."
  return {
    sentence: `The agent called ${asked(span)}. ${outcome}`,
    facts: facts(
      ["Tool", span.tool_name],
      ["Decision", decided],
      ["Took", span.outcome === "blocked" ? null : took(span)],
      ["Gates", gates ? describeTally(gates) : null],
    ),
  }
}

function llm(span: TraceSpan): Explanation {
  const reasons = span.attributes["gen_ai.response.finish_reasons"]
  const ttft = span.attributes["otari.llm.ttft_ms"]
  return {
    sentence: `${span.model ?? "The model"} ${span.outcome === "error" ? "failed" : "answered"}${took(span) ? ` in ${took(span)}` : ""}.`,
    facts: facts(
      ["Model", span.model],
      ["Input tokens", count(span.input_tokens)],
      ["Output tokens", count(span.output_tokens)],
      ["Cache read", count(span.cache_read_tokens)],
      ["Cache written", count(span.cache_write_tokens)],
      [
        "First token after",
        typeof ttft === "number" ? formatDuration(ttft) : null,
      ],
      [
        "Stopped because",
        typeof reasons === "string"
          ? reasons
              .split(",")
              .map((reason) => STOPPED_BECAUSE[reason] ?? reason)
              .join("; ")
          : null,
      ],
    ),
  }
}

function prompt(span: TraceSpan): Explanation {
  return {
    sentence: span.content.prompt
      ? "The prompt that started this turn."
      : "A turn the agent took. Its prompt is not kept here.",
    facts: facts(["Took", took(span)]),
  }
}

export function explainSpan(
  span: TraceSpan,
  parent: TraceSpan | undefined,
  blockedBy: readonly string[],
  gates: GateTally | undefined,
): Explanation | undefined {
  if (span.operation === "tool.blocked_on_user") return permission(span, parent)
  if (span.operation === "tool.execution") return ran(span)
  if (span.kind === "tool") return tool(span, blockedBy, gates)
  if (span.kind === "llm") return llm(span)
  if (span.kind === "step" && span.operation === "interaction")
    return prompt(span)
  return undefined
}

// The gates one tool call or one stop ran, by what they decided.
export type GateTally = Partial<Record<GuardrailVerdict, number>>

const TALLY_ORDER: [GuardrailVerdict, string][] = [
  ["blocked", "blocked"],
  ["refused", "could not decide"],
  ["unchecked", "could not decide"],
  ["flagged", "flagged"],
  ["passed", "passed"],
]

export function describeTally(tally: GateTally): string {
  const byWord = new Map<string, number>()
  for (const [verdict, word] of TALLY_ORDER) {
    const n = tally[verdict]
    if (n) byWord.set(word, (byWord.get(word) ?? 0) + n)
  }
  return Array.from(byWord, ([word, n]) => `${n} ${word}`).join(", ")
}

// A group's verdicts in words, or how many checks it holds when none says one.
export function groupWords(tally: GateTally, count: number): string {
  return describeTally(tally) || (count === 1 ? "1 check" : `${count} checks`)
}

export function gateGroupExplanation(
  tally: GateTally,
  atStop: boolean,
  count: number,
): Explanation {
  return {
    sentence: `${count === 1 ? "1 gate" : `${count} gates`} checked ${atStop ? "the turn when the agent stopped" : "this tool call before it ran"}: ${groupWords(tally, count)}. Open one to see what it is for.`,
    facts: [],
  }
}
