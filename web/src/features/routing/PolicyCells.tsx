import type { PolicySpec } from "@/client"
import { Chip } from "@/design-system/indicators/Chip"
import { Dot } from "@/design-system/indicators/Dot"
import { describeOffering } from "@/features/models/offerings"

import {
  computeShares,
  findBudgetConditions,
  findCandidates,
  findFallthroughTarget,
  findRouterBackend,
  findWeights,
  KNN_BACKEND,
  PRIORITY_BACKEND,
  type RoutingRow,
  WEIGHTED_BACKEND,
} from "./policyModel"

/**
 * What to call the backend that decides, for a mark or a one-line summary.
 *
 * Named per backend rather than "Dynamic", because the backend's name is what
 * tells the reader what to do next (teach it, or move the shares). A backend
 * this build does not know gets the neutral word.
 */
export function routerLabelOf(spec: PolicySpec): string {
  const backend = findRouterBackend(spec)
  if (backend === WEIGHTED_BACKEND) return "Weighted"
  if (backend === KNN_BACKEND) return "Learned"
  if (backend === PRIORITY_BACKEND) return "Priority"
  return "Routed"
}

/** A selector in a table cell: its model, then its provider. */
export function selectorText(selector: string): string {
  const { model, provider } = describeOffering(selector)
  return provider ? `${model} · ${provider}` : model
}

/** A routed policy's pool in one line: the split, or the pool and its default. */
function routedSummary(spec: PolicySpec, pool: string[]): string {
  const target = findFallthroughTarget(spec)
  if (findRouterBackend(spec) === WEIGHTED_BACKEND) {
    // The split's shape, not the model names: the shares are what tells one
    // weighted policy from another, and the pool is spelled out in the editor.
    const declared = findWeights(spec)
    const full = pool.includes(target) ? pool : [...pool, target]
    const split = computeShares(full.map((selector) => declared[selector] ?? 0))
      .map((share) => `${Math.round(share)}%`)
      .join(" / ")
    return `Weighted · ${split} across ${full.length} models`
  }
  return `${routerLabelOf(spec)} · ${pool.length} candidates, ${selectorText(target)} by default`
}

/** The affirmative kind mark: an accent dot and the word, in mono. */
function KindMark({ label }: { label: string }) {
  return (
    <span className="flex items-center gap-2 text-mono-caption text-foreground">
      <Dot className="bg-accent" />
      {label.toUpperCase()}
    </span>
  )
}

/** A selector as its model and provider, without a catalog read per row. */
function SelectorName({ selector }: { selector: string }) {
  const { model, provider } = describeOffering(selector)
  return (
    <span title={selector} className="flex min-w-0 items-baseline gap-1.5">
      <span className="text-sm text-foreground">{model}</span>
      {provider ? <span className="text-caption">{provider}</span> : null}
    </span>
  )
}

/**
 * What serves a normal request through this row, with its tier-downs and
 * guardrails each on a caption line of its own.
 */
export function MainModelCell({ row }: { row: RoutingRow }) {
  const pool = findCandidates(row.spec)
  const conditions = findBudgetConditions(row.spec)
  const guardrails = row.spec.guardrails ?? []
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        {pool.length > 0 ? (
          <span className="text-sm text-foreground">
            {routedSummary(row.spec, pool)}
          </span>
        ) : (
          <SelectorName selector={findFallthroughTarget(row.spec)} />
        )}
        {/* A router or a condition is a decision somebody made about this
            policy; a plain one-target policy is just the default shape. */}
        {pool.length > 0 ? (
          <KindMark label={routerLabelOf(row.spec)} />
        ) : row.is_dynamic ? (
          <KindMark label="Dynamic" />
        ) : null}
        {row.kind === "alias" ? (
          <span className="text-mono-overline text-subtle">alias</span>
        ) : null}
      </div>
      {conditions.map((condition) => (
        <span
          key={`${condition.threshold}:${condition.target}`}
          className="text-caption"
        >
          From {condition.threshold}% of budget:{" "}
          {selectorText(condition.target)}
        </span>
      ))}
      {guardrails.length > 0 ? (
        <span className="text-caption">
          Always checks{" "}
          {guardrails
            .map((guardrail) => `${guardrail.profile} (${guardrail.mode})`)
            .join(", ")}
        </span>
      ) : null}
    </div>
  )
}

/** The failure chain, in order, as chips. */
export function BackupsCell({ row }: { row: RoutingRow }) {
  const chain = row.spec.on_failure ?? []
  if (chain.length === 0) return <span className="text-muted">None</span>
  return (
    <ol aria-label="Backups" className="flex flex-wrap gap-1.5">
      {chain.map((selector, index) => (
        <li key={`${index}:${selector}`}>
          <Chip>
            <span title={selector}>{selectorText(selector)}</span>
          </Chip>
        </li>
      ))}
    </ol>
  )
}
