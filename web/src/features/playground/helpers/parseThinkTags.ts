// Splitting a reply's inline reasoning out of its visible body.
//
// Two ways a model reports chain-of-thought and the page shows them the same
// way: a separate `reasoning` field on the stream (handled by the reducers) and
// `<think>…</think>` inline in the content, handled here. A model uses one or
// the other, never both.

export interface ParsedThinkTags {
  /** The reasoning, joined, or undefined when the reply carried none. */
  thinking: string | undefined
  /** What is left to render as the answer. */
  response: string
}

const OPEN_TAG = "<think>"
const CLOSE_TAG = "</think>"

/**
 * Split an assistant reply into its inline reasoning and its visible response.
 *
 * **Streaming-safe, which is the whole difficulty.** While the reasoning is
 * still arriving, `<think>` has opened and `</think>` has not: everything after
 * the opening tag is treated as thinking, so partial reasoning and the raw tag
 * never flash into the visible answer and then disappear when a later chunk
 * closes them.
 *
 * Handles several blocks in one reply, because a turn that called a tool comes
 * back as one completion whose content holds a reasoning block before the call
 * and another after the result. Every block is stripped and their contents
 * joined, so a later block cannot leak into the body.
 */
export function parseThinkTags(content: string): ParsedThinkTags {
  const thinkingParts: string[] = []
  const responseParts: string[] = []
  let rest = content

  while (true) {
    const openIndex = rest.indexOf(OPEN_TAG)
    if (openIndex === -1) {
      responseParts.push(rest)
      break
    }

    responseParts.push(rest.slice(0, openIndex))
    const afterOpen = rest.slice(openIndex + OPEN_TAG.length)
    const closeIndex = afterOpen.indexOf(CLOSE_TAG)

    // Still streaming this block: no closing tag yet, so everything after the
    // opening one is thinking and nothing past it can be response.
    if (closeIndex === -1) {
      thinkingParts.push(afterOpen)
      break
    }

    thinkingParts.push(afterOpen.slice(0, closeIndex))
    rest = afterOpen.slice(closeIndex + CLOSE_TAG.length)
  }

  const thinking = thinkingParts
    .map((part) => part.trim())
    .filter(Boolean)
    .join("\n\n")
  return {
    thinking: thinking || undefined,
    response: responseParts.join("").trim(),
  }
}
