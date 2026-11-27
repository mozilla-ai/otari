import { Link } from "@tanstack/react-router"
import { Fragment } from "react"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import type { emptyCopy, ListEntry } from "@/features/rules/rules"

/** An allow-list in one line: words for a wildcard, ids in mono. */
export function EntryList({ entries }: { entries: readonly ListEntry[] }) {
  return (
    <span
      className="block truncate text-body"
      title={entries.map((one) => one.text).join(", ")}
    >
      {entries.map((one, index) => (
        <Fragment key={one.text}>
          {index > 0 ? ", " : null}
          {one.isId ? (
            <code className="text-mono-caption text-foreground">
              {one.text}
            </code>
          ) : (
            one.text
          )}
        </Fragment>
      ))}
    </span>
  )
}

/** One band with nothing in it, in a line, the page around it populated. */
export function BandEmpty({ copy }: { copy: ReturnType<typeof emptyCopy> }) {
  return (
    <EmptyMessage>
      <span className="flex flex-col gap-1">
        <span className="text-body">{copy.title}</span>
        {copy.body ? (
          <span>
            {copy.body}
            {copy.link ? (
              <>
                {" "}
                <Link to={copy.link.to} className="text-link underline">
                  {copy.link.label}
                </Link>
                .
              </>
            ) : null}
          </span>
        ) : null}
      </span>
    </EmptyMessage>
  )
}
