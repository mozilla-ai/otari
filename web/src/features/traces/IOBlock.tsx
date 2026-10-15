export type Format = "formatted" | "raw"

// Input or output of a trace or observation. Formatted reads as text, Raw is
// the JSON the record carries. No fill: the section title says which is which.
function tryJson(text: string): unknown {
  const trimmed = text.trim()
  if (!(trimmed.startsWith("{") || trimmed.startsWith("["))) return undefined
  try {
    return JSON.parse(trimmed)
  } catch {
    return undefined
  }
}

// One chat message: its text as prose, or its JSON when the content is
// structured.
function MessageContent({ message }: { message: unknown }) {
  const content = (message as { content?: unknown }).content
  const embedded = typeof content === "string" ? tryJson(content) : undefined
  if (typeof content === "string" && embedded === undefined) {
    return <p className="text-body whitespace-pre-wrap">{content}</p>
  }
  return (
    <pre className="overflow-x-auto text-mono-caption whitespace-pre-wrap">
      {JSON.stringify(embedded ?? message, null, 2)}
    </pre>
  )
}

export function IOBlock({
  label,
  value,
  format,
}: {
  label: string
  value: unknown
  format: Format
}) {
  if (value == null) return null
  // A string that is really JSON reads as an object in Formatted; Raw is as stored.
  const parsed =
    format === "formatted" && typeof value === "string"
      ? tryJson(value)
      : undefined
  if (parsed !== undefined) value = parsed
  const messages = Array.isArray(value)
    ? value
    : value && typeof value === "object" && "role" in (value as object)
      ? [value]
      : null
  const rawText =
    value && typeof value === "object" && "message" in (value as object)
      ? String((value as { message: unknown }).message)
      : null
  const embedded =
    format === "formatted" && rawText !== null ? tryJson(rawText) : undefined
  if (embedded !== undefined) value = embedded
  const text = embedded !== undefined ? null : rawText
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-overline">{label}</h3>
      {format === "formatted" && messages ? (
        <div className="flex flex-col gap-2">
          {messages.map((message, index) => (
            <div key={index} className="flex flex-col gap-0.5">
              <span className="text-mono-caption text-muted">
                {String((message as { role?: string }).role ?? "message")}
              </span>
              <MessageContent message={message} />
            </div>
          ))}
        </div>
      ) : format === "formatted" && text ? (
        <p className="text-body whitespace-pre-wrap">{text}</p>
      ) : (
        <pre className="overflow-x-auto text-mono-caption whitespace-pre-wrap">
          {JSON.stringify(value, null, 2)}
        </pre>
      )}
    </section>
  )
}
