import {
  Children,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from "react"
import { FiRotateCcw } from "react-icons/fi"
import type { Components } from "react-markdown"
import { Button } from "@/design-system/actions/Button"
import { CopyButton } from "@/design-system/actions/CopyButton"
import { CodeBlock } from "@/design-system/content/CodeBlock"
import { Markdown } from "@/design-system/content/Markdown"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { parseThinkTags } from "./helpers/parseThinkTags"
import { formatTurnStats } from "./helpers/playgroundCost"
import { splitModelKey } from "./helpers/playgroundModels"
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

export function MessageBubble({
  turn,
  model,
  onRegenerate,
  areActionsVisible = true,
}: {
  model: string
  turn: ChatTurn
  /** Given only on the latest finished answer. */
  onRegenerate?: () => void
  /** Hidden while this turn is the reply still streaming in. */
  areActionsVisible?: boolean
}) {
  if (turn.role === "user") {
    return (
      <div className="flex flex-col gap-2 border-b border-border pb-6">
        <p className="text-overline">You</p>
        <p className="whitespace-pre-wrap break-words text-base leading-[1.625rem]">
          {turn.content}
        </p>
      </div>
    )
  }

  const { thinking: inlineThinking, response } = parseThinkTags(turn.content)
  // A model streams reasoning in a field of its own or inline in `<think>`
  // tags, never both, so whichever arrived goes in the same block.
  const reasoning = turn.reasoning ?? inlineThinking
  const identity = splitModelKey(model)

  return (
    <div className="flex min-w-0 flex-col gap-4 break-words">
      <div className="flex flex-wrap items-baseline gap-2">
        <p className="text-overline">{identity.label}</p>
        {identity.instance ? (
          <span className="text-caption">{identity.instance}</span>
        ) : null}
      </div>
      {reasoning ? <ThinkingBlock content={reasoning} /> : null}
      {response ? (
        <Markdown components={ANSWER_MARKDOWN}>{response}</Markdown>
      ) : null}
      {turn.errorMessage ? (
        // Inline rather than a toast: a failure belongs where the answer would
        // have been, so a transcript still reads in order afterwards and a
        // failure scrolled past is still findable.
        <ErrorBanner error={new Error(turn.errorMessage)} />
      ) : null}
      {turn.usage ? (
        <p className="text-mono-caption tabular-nums text-muted">
          {formatTurnStats(turn.usage)}
        </p>
      ) : null}
      {/* Shown for a failed turn too, not only a successful one. A stream that
          died before its first token leaves `content` empty, so gating on the
          response alone hid the whole row at the one moment somebody wants
          Regenerate. Copy still needs something to copy. */}
      {areActionsVisible && (response || turn.errorMessage) ? (
        <div className="otari-actions flex flex-wrap items-center gap-2">
          {response ? (
            <CopyButton value={response} label="response" showLabel />
          ) : null}
          {onRegenerate ? (
            <Button aria-label="Regenerate response" onPress={onRegenerate}>
              <FiRotateCcw aria-hidden className="size-4" /> Regenerate
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
