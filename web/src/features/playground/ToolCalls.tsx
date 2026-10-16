import { FiTool } from "react-icons/fi"

import { CodeBlock } from "@/design-system/content/CodeBlock"
import { Disclosure } from "@/design-system/navigation/Disclosure"

import type { ToolCall } from "./helpers/playgroundTypes"

/**
 * The model's arguments, indented when they parse as JSON and verbatim when
 * they do not: a call cut off by Stop or a model that sent malformed JSON still
 * shows exactly what arrived.
 */
export function formatToolArguments(raw: string): string {
  if (!raw.trim()) return ""
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}

/**
 * The tools the gateway ran for one reply, each collapsed to a heading that
 * says what ran and how it went, opening to its arguments and its result.
 *
 * Collapsed because the answer is what somebody reads and the calls are how it
 * was reached; the status in the heading is what tells them to look.
 */
export function ToolCalls({
  calls,
  isStreaming,
}: {
  calls: readonly ToolCall[]
  /** Whether the reply is still arriving, which is what "waiting" means. */
  isStreaming: boolean
}) {
  if (calls.length === 0) return null
  return (
    <ul aria-label="Tool calls" className="flex flex-col">
      {calls.map((call, index) => (
        <li
          // Index as key: calls are append-only and an id can be empty.
          key={index}
          className="border-b border-border last:border-b-0"
        >
          <ToolCallRow call={call} isStreaming={isStreaming} />
        </li>
      ))}
    </ul>
  )
}

function ToolCallRow({
  call,
  isStreaming,
}: {
  call: ToolCall
  isStreaming: boolean
}) {
  const args = formatToolArguments(call.arguments)
  const hasResult = call.result !== undefined
  const status = hasResult
    ? call.isError
      ? "Failed"
      : "Done"
    : isStreaming
      ? "Running"
      : "No result"

  return (
    <Disclosure
      heading={
        <span className="flex min-w-0 items-center gap-2">
          <FiTool aria-hidden className="size-4 shrink-0 text-subtle" />
          <span className="truncate text-emphasis">{call.name || "Tool"}</span>
          <span
            className={`shrink-0 text-caption ${
              call.isError ? "text-danger" : ""
            }`}
          >
            {status}
          </span>
        </span>
      }
    >
      <div className="flex flex-col gap-3">
        {args ? <CodeBlock label="Arguments" value={args} /> : null}
        {hasResult ? (
          <div className="flex flex-col gap-1">
            <CodeBlock label="Result" value={call.result} isBounded />
            {call.isTruncated ? (
              <p className="text-caption">Result shortened for display.</p>
            ) : null}
          </div>
        ) : null}
      </div>
    </Disclosure>
  )
}
