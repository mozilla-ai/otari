import type { TraceSpan } from "@/client"
import { docsSourceHref } from "@/shared/helpers/docs"

import { CONTENT_PRIVATE } from "./exceptionalAccess"
import { formatDuration } from "./format"
import { type GuardrailVerdict, gateVerdict } from "./gateway"
import { clip, clipPath } from "./traceModel"

// What a span means, in a sentence and a few facts, for the panel a picked span
// opens. Built from what the span already holds; nothing here is stored.
export interface Explanation {
  sentence: string
  facts: Fact[]
}

/**
 * An MCP server's page, in the workspace the call ran in, with its access test
 * set to one key and tool when both are known.
 */
export interface McpServerLink {
  serverId: string
  workspace?: string
  key?: string
  tool?: string
}

export type Fact = [term: string, value: string, link?: McpServerLink]

// Why a span shows no input. The dashboard cannot tell a deployment that keeps
// no content from an agent that sent none or a reader not allowed to see it, so
// a span that can hold content says it is not shown, and links the docs on all three.
export interface NoInput {
  sentence: string
  docsHref?: string
}

export const KEEPING_CONTENT = docsSourceHref(
  "sessions.md",
  "keep-what-a-session-was-about",
)

// The tools Claude Code reports a command or a file path for. Otari keeps
// Claude Code's full_command and file_path; these are the tools that send them.
const TOOL_KEEPS: Record<string, string> = {
  Bash: "command",
  Read: "file path",
  Edit: "file path",
  Write: "file path",
}

// A step that opens a turn can keep its prompt, routed or exported. A routed
// request can keep each tool call's arguments; a tool Claude Code exported only
// a command or file path, and an exported gate what it checked.
function notShown(span: TraceSpan): string | undefined {
  if (span.kind === "step" && span.opens_turn) return "Its prompt is not shown."
  if (span.origin === "gateway")
    return span.kind === "tool" ? "Its arguments are not shown." : undefined
  const tool = span.tool_name ?? ""
  // Claude Code names its tool spans' operation "tool"; other exporters' tools keep nothing.
  if (
    span.kind === "tool" &&
    span.operation === "tool" &&
    Object.hasOwn(TOOL_KEEPS, tool)
  )
    return `Its ${TOOL_KEEPS[tool]} is not shown.`
  if (gateVerdict(span)) return "What the gate checked is not shown."
  return undefined
}

export const TURN_NO_INPUT: NoInput = {
  sentence: "A turn keeps nothing of its own. Its steps are under it.",
}

export const GATES_NO_INPUT: NoInput = { sentence: "Its gates are under it." }

// A span holding content this reader may not open: who may, rather than why there is none.
export const PRIVATE_NO_INPUT: NoInput = { sentence: CONTENT_PRIVATE }

export function noInput(span: TraceSpan): NoInput {
  const sentence = notShown(span)
  if (sentence) return { sentence, docsHref: KEEPING_CONTENT }
  if (span.kind === "llm")
    return {
      sentence: "Otari never keeps what a model call sent or got back.",
    }
  if (span.kind === "step")
    return {
      sentence: "A request that continues a turn keeps nothing of its own.",
    }
  if (span.kind === "tool")
    return { sentence: "Otari keeps no content for this tool." }
  return { sentence: "Otari keeps no content for this kind of span." }
}

const DECIDED_BY: Record<string, string> = {
  config: "a permission rule in your settings",
  hook: "a hook",
  user: "the user",
  mcp_key: "the key's rule on this MCP server",
  mcp_user: "the user's rule on this MCP server",
  mcp_everyone: "this MCP server's default policy",
  mcp_server: "this MCP server's tool list",
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

function facts(
  ...pairs: [string, string | null | undefined, McpServerLink?][]
): Fact[] {
  return pairs.filter((pair): pair is Fact => Boolean(pair[1]))
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

function mcpServer(span: TraceSpan): string | null {
  const server = span.attributes["otari.mcp.server"]
  return typeof server === "string" ? server : null
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
  apiKeyId: string | null,
  workspaceId: string | null,
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
  const serverId = span.attributes["otari.mcp.server_id"]
  const server =
    typeof serverId === "string"
      ? { serverId, workspace: workspaceId ?? undefined }
      : undefined
  return {
    sentence: `The agent called ${asked(span)}. ${outcome}`,
    facts: facts(
      ["Tool", span.tool_name],
      ["MCP server", mcpServer(span), server],
      [
        "Decision",
        decided,
        server && apiKeyId && span.tool_name
          ? { ...server, key: apiKeyId, tool: span.tool_name }
          : server,
      ],
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
      : "A turn the agent took.",
    facts: facts(["Took", took(span)]),
  }
}

export function explainSpan(
  span: TraceSpan,
  parent: TraceSpan | undefined,
  blockedBy: readonly string[],
  gates: GateTally | undefined,
  apiKeyId: string | null = null,
  workspaceId: string | null = null,
): Explanation | undefined {
  if (span.operation === "tool.blocked_on_user") return permission(span, parent)
  if (span.operation === "tool.execution") return ran(span)
  if (span.kind === "tool")
    return tool(span, blockedBy, gates, apiKeyId, workspaceId)
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
