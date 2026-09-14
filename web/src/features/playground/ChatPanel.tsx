import { Skeleton } from "@/design-system/feedback/Skeleton"

import type { PanelState } from "./helpers/playgroundTypes"
import { MessageBubble } from "./MessageBubble"

/**
 * One model's side of the conversation.
 *
 * A list and nothing else: no scroll container of its own, which is the
 * deliberate part. The shell already scrolls the page, and a transcript that
 * scrolled inside it would put a second scrollbar inside the first, which is
 * both worse on a phone and impossible to reason about while comparing (two
 * inner scrollbars beside each other, neither of which is the one the wheel is
 * over). Following the conversation down is the page's job for the same reason:
 * there is one scroll position, so there is one place to decide about it, and
 * `PlaygroundPage` owns it.
 */
export function ChatPanel({
  panel,
  isBordered = false,
  onRegenerate,
}: {
  panel: PanelState
  /** Frame the panel. Used while comparing, to separate the two columns. */
  isBordered?: boolean
  onRegenerate: () => void
}) {
  return (
    <div
      className={`flex min-w-0 flex-1 flex-col gap-4 ${
        isBordered ? "rounded-lg border border-border p-4" : ""
      }`}
    >
      {panel.turns.length === 0 && !panel.isAwaitingFirstToken ? (
        <p className="text-caption">Send a message to start.</p>
      ) : null}

      {panel.turns.map((turn, index) => {
        const isLast = index === panel.turns.length - 1
        return (
          <MessageBubble
            // Index as key, deliberately. A turn has no id (nothing has stored
            // it yet) and the list is append-only with the trailing turn
            // patched in place, so the index *is* the identity: a
            // content-derived key would change on every streamed fragment and
            // remount the bubble, losing its copy state on each token.
            key={index}
            turn={turn}
            // Hidden on the reply still streaming: a Copy of a half-answer and
            // a Regenerate of a request in flight are both wrong.
            areActionsVisible={!(isLast && panel.isStreaming)}
            onRegenerate={
              isLast && turn.role === "assistant" && !panel.isStreaming
                ? onRegenerate
                : undefined
            }
          />
        )
      })}

      {panel.isAwaitingFirstToken ? (
        // A skeleton rather than a spinner: it occupies the space the answer is
        // about to fill, so the first token does not shift the page.
        <Skeleton className="h-16 w-full" />
      ) : null}
    </div>
  )
}
