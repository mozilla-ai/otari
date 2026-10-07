export type Format = "formatted" | "raw"

// Input or output of a trace or observation. Formatted reads as text, Raw is
// the JSON the record carries. No fill: the section title says which is which.
export function IOBlock({
  label,
  value,
  format,
}: {
  label: string
  value: unknown
  format: Format
}) {
  const messages = Array.isArray(value)
    ? value
    : value && typeof value === "object" && "role" in (value as object)
      ? [value]
      : null
  const text =
    value && typeof value === "object" && "message" in (value as object)
      ? String((value as { message: unknown }).message)
      : null
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-overline text-muted">{label}</h3>
      {value == null ? (
        <span className="text-subtle">—</span>
      ) : format === "formatted" && messages ? (
        <div className="flex flex-col gap-2">
          {messages.map((message, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: static mock content
            <div key={index} className="flex flex-col gap-0.5">
              <span className="text-mono-caption text-muted">
                {String((message as { role?: string }).role ?? "message")}
              </span>
              {typeof (message as { content?: unknown }).content ===
              "string" ? (
                <p className="text-body whitespace-pre-wrap">
                  {(message as { content: string }).content}
                </p>
              ) : (
                <pre className="overflow-x-auto text-mono-caption whitespace-pre-wrap">
                  {JSON.stringify(message, null, 2)}
                </pre>
              )}
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
