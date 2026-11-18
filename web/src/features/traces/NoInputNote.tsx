import { DocsLink } from "@/design-system/navigation/DocsLink"

import type { NoInput } from "./explain"

export function NoInputNote({
  noInput,
  className = "",
}: {
  noInput: NoInput
  className?: string
}) {
  return (
    <p className={`text-caption ${className}`}>
      {noInput.sentence}
      {noInput.docsHref ? (
        <>
          {" "}
          <DocsLink href={noInput.docsHref}>How to keep it</DocsLink>
        </>
      ) : null}
    </p>
  )
}
