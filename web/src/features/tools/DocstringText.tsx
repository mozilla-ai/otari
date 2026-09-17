import { Fragment, type ReactNode } from "react"

// A guardrail's parameter help, as it reads in the language it was written in.
//
// Every description on the catalog is the first lines of a Python docstring, so
// it arrives in reStructuredText: a literal is wrapped in double backticks. Put
// straight into a form that renders as ``all_pass``, and a form full of stray
// backticks reads as a bug in the form rather than as markup nobody rendered.
//
// Only literals are handled, deliberately. They are the one construct these
// descriptions actually use, and a general renderer would be a Markdown
// dependency for text the gateway already trims to a line or two.

/** Splits on ``literals``, keeping them. An unclosed pair stays as typed. */
const LITERAL = /``([^`]+)``/g

// Ends a sentence, but not inside a literal and not on an abbreviation or a
// decimal. Matched on the period that is followed by a space and a capital,
// which is what separates "0.8 to tune it." from "e.g. ``{...}``".
const SENTENCE_END = /\.\s+(?=[A-Z(])/

/**
 * The first sentence, which is the line a field has room for.
 *
 * These descriptions are a docstring's argument paragraph, so the first sentence
 * says what the argument is and the rest says what to pass, in the vocabulary of
 * the Python call rather than of the form. What a value may be is the vendor
 * reference's job, and the picker links to it.
 *
 * Trimming rather than rewriting, because the words are upstream's: a paraphrase
 * kept here would drift from a description that changed and nothing would say so.
 */
export function firstSentence(text: string): string {
  // Counted across the literals rather than through them, so a period inside
  // ``{"a": 0.8}`` cannot end the sentence.
  const guarded = text.replace(LITERAL, (match) =>
    "\u0000".repeat(match.length),
  )
  const at = guarded.search(SENTENCE_END)
  return at === -1 ? text.trim() : text.slice(0, at + 1).trim()
}

export function DocstringText({
  children,
  full,
}: {
  children: string | undefined
  /** Keep every sentence. For a place with room, such as the JSON view. */
  full?: boolean
}) {
  if (!children) return null
  const text = full ? children : firstSentence(children)
  const parts: ReactNode[] = []
  let at = 0
  for (const match of text.matchAll(LITERAL)) {
    const start = match.index
    if (start > at) parts.push(text.slice(at, start))
    parts.push(
      <code key={start} className="text-mono-caption">
        {match[1]}
      </code>,
    )
    at = start + match[0].length
  }
  if (at < text.length) parts.push(text.slice(at))
  return (
    <>
      {parts.map((part, index) => (
        // Keyed by position: the parts are an ordered split of one string and
        // nothing in them is an identity.
        <Fragment key={index}>{part}</Fragment>
      ))}
    </>
  )
}
