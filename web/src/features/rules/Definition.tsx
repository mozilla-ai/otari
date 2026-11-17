import { useId, useState } from "react"
import { FiChevronRight } from "react-icons/fi"
import { CodeBlock } from "@/design-system/content/CodeBlock"

/**
 * What the control plane stores for a rule, folded: one record per person
 * under a per-person rule runs to thousands of pixels, and it is the part of
 * the page read least.
 */
export function Definition({
  name,
  definition,
}: {
  name: string
  definition: unknown
}) {
  const [isOpen, setOpen] = useState(false)
  const panel = useId()
  const records = Array.isArray(definition) ? definition.length : 1
  return (
    <section className="flex flex-col gap-3 border-y border-border">
      <button
        type="button"
        aria-expanded={isOpen}
        aria-controls={panel}
        onClick={() => setOpen((open) => !open)}
        className="flex min-h-11 items-center gap-3 text-left"
      >
        <FiChevronRight
          aria-hidden
          className={`size-4 shrink-0 text-muted transition-transform motion-reduce:transition-none ${isOpen ? "rotate-90" : ""}`}
        />
        <span className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-0.5">
          <span className="text-title">Definition</span>
          <span className="text-caption">
            What the control plane stores for this rule, as JSON: {records}{" "}
            {records === 1 ? "record" : "records"}
          </span>
        </span>
      </button>
      {isOpen ? (
        <div id={panel} className="pb-4">
          <CodeBlock
            label={`${name}.json`}
            value={JSON.stringify(definition, null, 2)}
            isBounded
          />
        </div>
      ) : null}
    </section>
  )
}
