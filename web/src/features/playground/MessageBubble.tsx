import {
  Children,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from "react"
import { FiRotateCcw } from "react-icons/fi"
import type { Components } from "react-markdown"

import { CopyButton } from "@/design-system/actions/CopyButton"
import { IconButton } from "@/design-system/actions/IconButton"
import { CodeBlock } from "@/design-system/content/CodeBlock"
import { Markdown } from "@/design-system/content/Markdown"
import { Tooltip } from "@/design-system/overlays/Tooltip"

import { parseThinkTags } from "./helpers/parseThinkTags"
import { formatTurnStats } from "./helpers/playgroundCost"
import type { ChatTurn } from "./helpers/playgroundTypes"
import { ThinkingBlock } from "./ThinkingBlock"

/**
 * A fenced block in a model's answer, routed through the shared `CodeBlock`.
 *
 * The same shape the docs page uses, and for the same two reasons: the language
 * lives on the inner `<code>`'s class where remark puts it, and the text a copy
 * yields is that child's own string rather than the rendered nodes.
 *
 * It matters more here than on a page of hand-written prose. A model asked for
 * code answers with code, and the reader's next action is to copy it, so a fence
 * with no copy control is the most common action on this page left undone.
 */
function AnswerCodeBlock({ children }: { children?: ReactNode }) {
  const child = Children.toArray(children).find(isValidElement) as
    | ReactElement<{ className?: string; children?: ReactNode }>
    | undefined
  const language =
    /language-([\w+-]+)/.exec(child?.props.className ?? "")?.[1] ?? ""
  const text =
    typeof child?.props.children === "string" ? child.props.children : ""

  return (
    <CodeBlock
      label={language || undefined}
      value={text || undefined}
      // Not `isBounded`, which is a fixed `h-56` band: a one-line snippet in it
      // is a line of code above 200px of empty block, which is what it looked
      // like. A fence here sizes to its content and the page scrolls.
      className="my-3"
    >
      {children}
    </CodeBlock>
  )
}

const ANSWER_MARKDOWN: Components = {
  pre: ({ children }) => <AnswerCodeBlock>{children}</AnswerCodeBlock>,
  // A link in a model's answer opens in a new tab, so a stray click never
  // navigates away from a conversation that is not saved yet, and the opened
  // page gets no handle on this one.
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
}

/**
 * One turn of the conversation.
 *
 * A question is a bubble on the right, because it is short and the reader wrote
 * it. An answer is full width with no bubble at all: it is the thing being read,
 * often long and often containing a code fence, and a tinted container around
 * it both narrows the measure and fights the fence's own frame.
 */
export function MessageBubble({
  turn,
  onRegenerate,
  areActionsVisible = true,
}: {
  turn: ChatTurn
  /** Given only on the latest finished answer. */
  onRegenerate?: () => void
  /** Hidden while this turn is the reply still streaming in. */
  areActionsVisible?: boolean
}) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-lg bg-primary-subtle px-4 py-3 text-primary-subtle-foreground">
          <p className="whitespace-pre-wrap text-sm">{turn.content}</p>
        </div>
      </div>
    )
  }

  const { thinking: inlineThinking, response } = parseThinkTags(turn.content)
  // A model streams reasoning in a field of its own or inline in `<think>`
  // tags, never both, so whichever arrived goes in the same block.
  const reasoning = turn.reasoning ?? inlineThinking

  return (
    <div className="group flex flex-col gap-1">
      {reasoning ? <ThinkingBlock content={reasoning} /> : null}
      {response ? (
        <Markdown components={ANSWER_MARKDOWN}>{response}</Markdown>
      ) : null}
      {turn.errorMessage ? (
        // Inline rather than a toast: a failure belongs where the answer would
        // have been, so a transcript still reads in order afterwards and a
        // failure scrolled past is still findable.
        <p className="text-sm text-danger">{turn.errorMessage}</p>
      ) : null}
      {turn.usage ? (
        <p className="text-caption">{formatTurnStats(turn.usage)}</p>
      ) : null}
      {/* Shown for a failed turn too, not only a successful one. A stream that
          died before its first token leaves `content` empty, so gating on the
          response alone hid the whole row at the one moment somebody wants
          Regenerate. Copy still needs something to copy. */}
      {areActionsVisible && (response || turn.errorMessage) ? (
        // Always visible on a touch screen and revealed on hover from `md` up:
        // a hover-only control is unreachable on a phone, which the
        // responsiveness rule forbids outright.
        <div className="flex items-center gap-1 opacity-100 transition-opacity md:opacity-0 md:group-hover:opacity-100 md:group-focus-within:opacity-100">
          {response ? (
            <CopyButton value={response} label="Copy response" />
          ) : null}
          {onRegenerate ? (
            <Tooltip content="Regenerate">
              <IconButton label="Regenerate response" onPress={onRegenerate}>
                <FiRotateCcw aria-hidden className="size-4" />
              </IconButton>
            </Tooltip>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
