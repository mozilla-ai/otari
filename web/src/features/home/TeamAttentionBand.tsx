import { Link } from "@tanstack/react-router"
import { useState } from "react"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { Dot } from "@/design-system/indicators/Dot"
import {
  type Severity,
  SeverityMark,
} from "@/design-system/metrics/SeverityMark"
import type { AttentionItem } from "@/features/home/attention"
import { useTeamAttention } from "@/features/home/useTeamAttention"
import type { LimitTarget } from "@/features/spend/limits"
import type { Spend } from "@/features/spend/useSpend"

const ROW =
  "flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-border-subtle py-2.5"

/** How urgent an item is, in the band's word for it. */
export const severityOf = (isUrgent: boolean): Severity =>
  isUrgent
    ? { status: "alert", word: "Urgent" }
    : { status: "warn", word: "Warning" }

/**
 * One line of the band; a member's band (`MemberAttentionBand.tsx`) too. An
 * item that needs someone carries its severity as a word, the rest a dot.
 */
export function Row({
  dot,
  severity,
  children,
  action,
}: {
  dot?: string
  severity?: Severity
  children: React.ReactNode
  action?: React.ReactNode
}) {
  return (
    <li className={ROW}>
      {severity ? (
        <span className="flex min-w-0 flex-1 basis-64 items-start gap-2.5 text-sm">
          {/* One width for both words, so the sentences start in one column. */}
          <span className="w-20 shrink-0 text-mono-caption leading-5">
            <SeverityMark severity={severity} />
          </span>
          <span>{children}</span>
        </span>
      ) : (
        <span className="flex min-w-0 flex-1 basis-64 items-baseline gap-2.5 text-sm">
          <Dot className={`relative -top-0.5 ${dot ?? ""}`} />
          <span>{children}</span>
        </span>
      )}
      {action}
    </li>
  )
}

function ItemRow({
  item,
  onSetLimit,
}: {
  item: AttentionItem
  onSetLimit: (target: LimitTarget) => void
}) {
  const { action } = item
  const { limit } = action
  return (
    <Row
      severity={severityOf(item.isUrgent)}
      action={
        limit ? (
          <button
            type="button"
            onClick={() => onSetLimit(limit)}
            className="shrink-0 text-sm text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
          >
            {action.label}
          </button>
        ) : action.node ? (
          action.node
        ) : action.to ? (
          <Link
            to={action.to}
            search={action.search}
            className="shrink-0 text-sm text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
          >
            {action.label}
          </Link>
        ) : null
      }
    >
      {item.text}
    </Row>
  )
}

/**
 * What needs an owner or admin, for the whole organization and the last 24
 * hours: the rules are in `attention.ts`. It keeps its height while it reads,
 * so the page below does not jump when it answers.
 */
export function TeamAttentionBand({
  spend,
  now,
  onSetLimit,
}: {
  spend: Spend
  now: number
  onSetLimit: (target: LimitTarget) => void
}) {
  const result = useTeamAttention(spend, now)
  const [isExpanded, setExpanded] = useState(false)
  const shown = isExpanded ? [...result.items, ...result.folded] : result.items
  return (
    <section aria-label="Needs attention" className="flex flex-col">
      <h2 className="pb-2 text-overline">Needs attention</h2>
      <ul className="flex flex-col border-t border-border">
        {result.isLoading ? (
          <li className={ROW} aria-busy="true">
            <Skeleton className="h-5 w-[28rem] max-w-full" />
          </li>
        ) : (
          <>
            {result.quiet ? (
              <Row dot="bg-success">
                {result.quiet.lead}{" "}
                <span className="text-muted">{result.quiet.detail}</span>
              </Row>
            ) : null}
            {shown.map((item) => (
              <ItemRow key={item.id} item={item} onSetLimit={onSetLimit} />
            ))}
            {result.folded.length > 0 ? (
              <li className={`${ROW} justify-between`}>
                {/* Expanded, every limit has its own row: no dot left to mark. */}
                <span className="min-w-0 flex-1 basis-64 pl-[1.125rem] text-sm text-muted">
                  {isExpanded ? null : result.foldText}
                </span>
                <button
                  type="button"
                  onClick={() => setExpanded((was) => !was)}
                  aria-expanded={isExpanded}
                  className="shrink-0 text-sm text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
                >
                  {isExpanded ? "Show fewer" : "Show all"}
                </button>
              </li>
            ) : null}
            {result.notChecked.map((text) => (
              <Row key={text} dot="bg-text-subtle">
                {text}
              </Row>
            ))}
          </>
        )}
      </ul>
    </section>
  )
}
