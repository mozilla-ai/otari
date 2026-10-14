import { Link } from "@tanstack/react-router"

import { CodeBlock } from "@/design-system/content/CodeBlock"

import { routerLabelOf, selectorText } from "./PolicyCells"
import { PolicyGroup } from "./PolicyGroup"
import {
  buildInitialPool,
  computeShares,
  findBudgetConditions,
  findFallthroughTarget,
  findRouterBackend,
  findWeights,
  type RoutingRow,
  WEIGHTED_BACKEND,
} from "./policyModel"

function NoneSet() {
  return <p className="text-sm text-muted">None</p>
}

/**
 * A policy this page will not edit, section by section, with the stored spec
 * in full beneath: the sections cover what the dashboard models, and the spec
 * is the whole of it.
 */
export function PolicyReadOnly({ row }: { row: RoutingRow }) {
  const { spec } = row
  const pool = buildInitialPool(spec)
  const target = findFallthroughTarget(spec)
  const shares =
    findRouterBackend(spec) === WEIGHTED_BACKEND
      ? computeShares(pool.map((selector) => findWeights(spec)[selector] ?? 0))
      : []
  const backups = spec.on_failure ?? []
  const tierDowns = findBudgetConditions(spec)
  const guardrails = spec.guardrails ?? []

  return (
    <>
      <PolicyGroup title="Main selection" isReadOnly>
        {pool.length === 0 ? (
          <p className="text-sm text-foreground">{selectorText(target)}</p>
        ) : (
          <div className="flex flex-col gap-2">
            <p className="text-caption">{routerLabelOf(spec)} router</p>
            <ol className="flex flex-col gap-1 text-sm text-foreground">
              {pool.map((selector, index) => (
                <li key={selector}>
                  {selectorText(selector)}
                  {shares.length > 0 ? (
                    <span className="text-muted">
                      {" "}
                      · {Math.round(shares[index] ?? 0)}%
                    </span>
                  ) : null}
                  {selector === target ? (
                    <span className="text-muted"> · default</span>
                  ) : null}
                </li>
              ))}
            </ol>
          </div>
        )}
      </PolicyGroup>
      <PolicyGroup title="Fallback chain" isReadOnly>
        {backups.length === 0 ? (
          <NoneSet />
        ) : (
          <ol className="flex list-decimal flex-col gap-1 pl-5 text-sm text-foreground">
            {backups.map((selector, index) => (
              <li key={`${index}:${selector}`}>{selectorText(selector)}</li>
            ))}
          </ol>
        )}
      </PolicyGroup>
      <PolicyGroup title="Budget tier-down" isReadOnly>
        {tierDowns.length === 0 ? (
          <NoneSet />
        ) : (
          <ul className="flex flex-col gap-1 text-sm text-foreground">
            {tierDowns.map((tier) => (
              <li key={`${tier.threshold}:${tier.target}`}>
                From {tier.threshold}% of budget: {selectorText(tier.target)}
              </li>
            ))}
          </ul>
        )}
      </PolicyGroup>
      <PolicyGroup title="Guardrails" isReadOnly>
        {guardrails.length === 0 ? (
          <NoneSet />
        ) : (
          <ul className="flex flex-col gap-1 text-sm text-foreground">
            {guardrails.map((guardrail) => (
              <li key={guardrail.profile}>
                {guardrail.profile} ({guardrail.mode}, if the service is down:{" "}
                {guardrail.on_unavailable ?? "block"})
              </li>
            ))}
          </ul>
        )}
        <p className="text-caption">
          Profiles and the service are managed on the{" "}
          <Link to="/tools/guardrails" className="text-link hover:underline">
            Guardrails page
          </Link>
          .
        </p>
      </PolicyGroup>
      <PolicyGroup title="Stored spec" isReadOnly>
        <CodeBlock label="json" value={JSON.stringify(spec, null, 2)} />
      </PolicyGroup>
    </>
  )
}
