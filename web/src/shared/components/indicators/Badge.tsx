/**
 * A small status pill for a row that has something to say about itself.
 *
 * Two tones and no more: `muted` states a fact about the row (which provider,
 * which scope, whether a credential is set), `warn` says the row is not doing
 * what its neighbors are. Lives here rather than in a feature folder because
 * the Tools cards each grew an identical copy.
 */
export function Badge({
  tone,
  children,
}: {
  tone: "muted" | "warn"
  children: string
}) {
  // The status form, not a pill: a square dot and a word. A `warn` badge says
  // the row is not doing what its neighbors are, which is what the danger dot
  // with muted words means everywhere else here; `muted` states a fact about
  // the row and takes the quiet dot.
  const dot = tone === "warn" ? "bg-danger" : "bg-text-subtle"
  const ink = tone === "warn" ? "text-muted" : "text-subtle"
  return (
    // Not uppercased and not mono, unlike the status marks it shares a dot
    // with. Those draw from a fixed vocabulary of one or two words; this one
    // takes whatever a caller passes, and its callers pass sentences ("Every
    // workspace, including new ones"). Uppercasing a sentence is shouting, and
    // mono is for values you copy rather than prose you read.
    <span className={`flex items-center gap-2 text-xs ${ink}`}>
      <span aria-hidden className={`h-1.5 w-1.5 shrink-0 ${dot}`} />
      {children}
    </span>
  )
}
