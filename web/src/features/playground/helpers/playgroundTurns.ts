// Reducers over a panel's turn list, and the scroll rule that reads them.
//
// Pure on purpose. Every one of these runs while a reply is streaming, twice
// over in split view, and they are the part of the page most likely to be wrong
// in a way nothing shows: a fold that mutates its input makes two panels share
// a turn, and a scroll rule that always follows makes a long answer unreadable
// because it yanks the view down on every token. Both are unit-testable here
// and neither is testable through a render.

import type {
  PlaygroundWireMessage,
  StreamDelta,
  ToolActivity,
} from "@/shared/api/playground"

import type { ChatTurn, ToolCall } from "./playgroundTypes"

/**
 * A copy of `turns` with its trailing assistant turn replaced by `patch(last)`.
 *
 * A fresh array always, and never a mutation, which is what lets two split-view
 * panels stream at once without one's reducer being visible in the other's
 * state. When the last turn is not an assistant's the list comes back unchanged
 * (still copied), so a patch that arrives after a reset is a no-op rather than a
 * write onto somebody's question.
 */
export function patchLastAssistantTurn(
  turns: ChatTurn[],
  patch: (last: ChatTurn) => ChatTurn,
): ChatTurn[] {
  const next = turns.slice()
  const last = next[next.length - 1]
  if (last?.role === "assistant") {
    next[next.length - 1] = patch(last)
  }
  return next
}

/**
 * Fold one streamed fragment in: append to the assistant turn in flight, or
 * start one on the first token.
 *
 * Content and reasoning accumulate independently, because a model that streams
 * both interleaves them and the two are rendered in different places. A tool's
 * activity joins the turn's tool calls, which render above the answer.
 */
export function appendStreamDelta(
  turns: ChatTurn[],
  delta: StreamDelta,
): ChatTurn[] {
  const next = turns.slice()
  const last = next[next.length - 1]
  const base: ChatTurn =
    last?.role === "assistant" ? last : { role: "assistant", content: "" }
  const reasoning = (base.reasoning ?? "") + (delta.reasoning ?? "")
  const updated: ChatTurn = {
    ...base,
    content: base.content + (delta.content ?? ""),
    reasoning: reasoning || undefined,
  }
  if (delta.toolActivity) {
    updated.toolCalls = foldToolActivity(
      base.toolCalls ?? [],
      delta.toolActivity,
    )
  }
  if (last?.role === "assistant") {
    next[next.length - 1] = updated
  } else {
    next.push(updated)
  }
  return next
}

/**
 * A copy of `calls` with one activity record folded in.
 *
 * A call starts a row, and its result fills the row it belongs to: by `id`, or,
 * when the model sent none, the earliest call of that name still waiting. A
 * result with no call to join (the call record was lost) still gets a row, so
 * what a tool returned is never dropped.
 */
export function foldToolActivity(
  calls: ToolCall[],
  activity: ToolActivity,
): ToolCall[] {
  if (activity.type === "tool_call") {
    return [
      ...calls,
      {
        id: activity.id,
        name: activity.name,
        arguments: activity.arguments,
        isError: false,
        isTruncated: false,
      },
    ]
  }
  const index = calls.findIndex((call) =>
    activity.id
      ? call.id === activity.id
      : call.name === activity.name && call.result === undefined,
  )
  const finished = {
    result: activity.content,
    isError: activity.isError,
    isTruncated: activity.isTruncated,
  }
  if (index === -1) {
    return [
      ...calls,
      { id: activity.id, name: activity.name, arguments: "", ...finished },
    ]
  }
  return calls.map((call, i) => (i === index ? { ...call, ...finished } : call))
}

/**
 * Surface a failure on the conversation rather than only in a toast.
 *
 * Attached to the partial reply when one had started, so a stream that died
 * halfway shows what arrived *and* why it stopped; a fresh turn otherwise. Kept
 * out of `content` and put in `errorMessage`, which is what lets the bubble
 * render it as a refusal instead of as something the model said, and keeps it
 * out of a saved transcript's text.
 */
export function appendErrorTurn(
  turns: ChatTurn[],
  errorMessage: string,
): ChatTurn[] {
  const next = turns.slice()
  const last = next[next.length - 1]
  if (last?.role === "assistant") {
    next[next.length - 1] = { ...last, errorMessage }
    return next
  }
  next.push({ role: "assistant", content: "", errorMessage })
  return next
}

/**
 * Truncate back to the last question so its answer can be regenerated.
 *
 * Returns the list unchanged when there is no question to re-answer, which the
 * caller reads as "nothing to regenerate" rather than having to check first.
 */
export function turnsForRegenerate(turns: ChatTurn[]): ChatTurn[] {
  const lastUser = turns.map((turn) => turn.role).lastIndexOf("user")
  if (lastUser === -1) return turns
  return turns.slice(0, lastUser + 1)
}

/**
 * Whether the transcript should follow new content to the bottom.
 *
 * A question the reader just sent always scrolls into view: they pressed send,
 * so the reply is what they are waiting for. Everything else follows only if
 * they were already at the bottom, which is what lets somebody scroll up to
 * read a long answer without the next token dragging them back down.
 */
export function shouldFollowToBottom(params: {
  hasNewTurn: boolean
  lastRole: string | undefined
  isPinnedToBottom: boolean
}): boolean {
  const justAsked = params.hasNewTurn && params.lastRole === "user"
  return justAsked || params.isPinnedToBottom
}

/**
 * The wire form of a panel's turns: what the model is sent, and nothing else.
 *
 * A question with attachments becomes content parts, its text first and then
 * one file part per attachment, which the gateway resolves by `file_id`.
 */
export function wireMessages(turns: ChatTurn[]): PlaygroundWireMessage[] {
  return turns.map((turn) => {
    if (!turn.attachments?.length) {
      return { role: turn.role, content: turn.content }
    }
    return {
      role: turn.role,
      content: [
        { type: "text", text: turn.content },
        ...turn.attachments.map((attachment) => ({
          type: "file" as const,
          file: { file_id: attachment.fileId, filename: attachment.filename },
        })),
      ],
    }
  })
}
