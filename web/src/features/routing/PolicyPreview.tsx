import { Fragment, useId } from "react"

import { offeringPhrase } from "@/features/models/offerings"
import { useSelectorOwner } from "@/features/models/useSelectorOwner"

import { KNN_BACKEND, WEIGHTED_BACKEND } from "./policyModel"

/** One target in words, named from the catalog once it answers. */
function Offering({ selector }: { selector: string }) {
  const owner = useSelectorOwner(selector)
  return (
    <span className="text-foreground">{offeringPhrase(selector, owner)}</span>
  )
}

/**
 * Where a policy sends a request, as one plain sentence under the form.
 *
 * Built from the draft in a form, so it changes as the operator types and says
 * what Save would do, and from the stored spec on the policy page.
 */
export function PolicyPreview({
  name,
  target,
  backups,
  pool,
  tierDowns,
  label = "Preview",
}: {
  name: string
  target: string
  backups: string[]
  /** The routed pool, when a router decides instead of one target. */
  pool?: { backend: string; size: number }
  tierDowns: { threshold: number; target: string }[]
  /** The heading; the policy page says what is in force rather than a preview. */
  label?: string
}) {
  const headingId = useId()
  const trimmedName = name.trim()
  const named = backups.map((entry) => entry.trim()).filter(Boolean)
  const conditions = tierDowns.filter((entry) => entry.target.trim() !== "")
  const subject =
    trimmedName === "" ? (
      "Requests"
    ) : (
      <>
        Requests to <code>{trimmedName}</code>
      </>
    )

  const sentence =
    pool === undefined && target.trim() === "" ? (
      <p className="text-caption">
        Pick a main model to see where requests go.
      </p>
    ) : (
      <p className="text-sm text-muted">
        {pool === undefined ? (
          <>
            {subject} go to <Offering selector={target} />.
          </>
        ) : pool.backend === WEIGHTED_BACKEND ? (
          <>
            {subject} are split across {pool.size} models by weight.
          </>
        ) : pool.backend === KNN_BACKEND ? (
          <>
            {subject} go to the cheapest of {pool.size} models a router expects
            to answer well.
          </>
        ) : (
          <>
            {subject} go to the first of {pool.size} models with room.
          </>
        )}
        {conditions.map((condition, index) => (
          // By position: a key that changed per keystroke would remount the
          // name and skip its debounce.
          <Fragment key={index}>
            {" "}
            Once {condition.threshold}% of the budget is used:{" "}
            <Offering selector={condition.target} />.
          </Fragment>
        ))}
        {named.length > 0 ? (
          <>
            {" "}
            If that fails:{" "}
            {named.map((entry, index) => (
              <Fragment key={index}>
                {index > 0 ? ", then " : ""}
                <Offering selector={entry} />
              </Fragment>
            ))}
            .
          </>
        ) : null}
      </p>
    )

  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-1">
      <span id={headingId} className="text-caption text-subtle">
        {label}
      </span>
      {sentence}
    </section>
  )
}
