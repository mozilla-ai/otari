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
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { parseThinkTags } from "./helpers/parseThinkTags"
import { splitModelKey } from "./helpers/playgroundModels"
import type { ChatTurn } from "./helpers/playgroundTypes"
import { ThinkingBlock } from "./ThinkingBlock"
import { TurnReadout } from "./TurnReadout"

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
      <article aria-label="Your message" className="flex justify-end">
        <p className="max-w-[35rem] whitespace-pre-wrap break-words bg-surface-alt px-4 py-3 text-base leading-[1.625rem]">
          {turn.content}
        </p>
      </article>
    )
  }

  const { thinking: inlineThinking, response } = parseThinkTags(turn.content)
  // A model streams reasoning in a field of its own or inline in `<think>`
  // tags, never both, so whichever arrived goes in the same block.
  const reasoning = turn.reasoning ?? inlineThinking
  const identity = splitModelKey(model)

  return (
    <div className="flex min-w-0 flex-col gap-3 break-words">
      <div className="flex flex-wrap items-baseline gap-2">
        <p className="text-emphasis">{identity.label}</p>
        {identity.instance ? (
          <span className="text-caption text-subtle">{identity.instance}</span>
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
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 pt-1">
        {areActionsVisible && (response || turn.errorMessage) ? (
          <div className="otari-actions flex shrink-0 items-center gap-3">
            {response ? <CopyButton value={response} label="response" /> : null}
            {onRegenerate ? (
              <IconButton
                isIconOnly
                size="sm"
                label="Regenerate response"
                className="md:min-h-8 md:min-w-8"
                onPress={onRegenerate}
              >
                <FiRotateCcw aria-hidden className="size-3.5" />
              </IconButton>
            ) : null}
          </div>
        ) : null}
        {turn.usage ? (
          <div className="ml-auto">
            <TurnReadout usage={turn.usage} />
          </div>
        ) : null}
      </div>
    </div>
  )
}
