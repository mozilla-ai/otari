// Reducers over a panel's turn list, and the scroll rule that reads them.
//
// Pure on purpose. Every one of these runs while a reply is streaming, twice
// over in split view, and they are the part of the page most likely to be wrong
// in a way nothing shows: a fold that mutates its input makes two panels share
// a turn, and a scroll rule that always follows makes a long answer unreadable
// because it yanks the view down on every token. Both are unit-testable here
// and neither is testable through a render.

import type { ChatTurn } from "./playgroundTypes"

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
 * both interleaves them and the two are rendered in different places.
 */
export function appendStreamDelta(
  turns: ChatTurn[],
  delta: { content?: string; reasoning?: string },
): ChatTurn[] {
  const next = turns.slice()
  const last = next[next.length - 1]
  if (last?.role === "assistant") {
    const reasoning = (last.reasoning ?? "") + (delta.reasoning ?? "")
    next[next.length - 1] = {
      ...last,
      content: last.content + (delta.content ?? ""),
      reasoning: reasoning || undefined,
    }
  } else {
    next.push({
      role: "assistant",
      content: delta.content ?? "",
      reasoning: delta.reasoning || undefined,
    })
  }
  return next
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

/** The wire form of a panel's turns: what the model is sent, and nothing else. */
export function wireMessages(
  turns: ChatTurn[],
): { role: string; content: string }[] {
  return turns.map((turn) => ({ role: turn.role, content: turn.content }))
}
