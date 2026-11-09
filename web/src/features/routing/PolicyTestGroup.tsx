import { useState } from "react"

import { Field } from "@/design-system/forms/Field"
import { useExplainPolicy } from "@/shared/api/routing"
import { useDebounced } from "@/shared/hooks/useDebounced"

import { selectorText } from "./PolicyCells"
import { PolicyGroup } from "./PolicyGroup"
import { findBudgetConditions, type RoutingRow } from "./policyModel"

/** Why a candidate is in the plan, in the words the sections above use. */
function reasonLabel(reason: string): string {
  if (reason === "default") return "main model"
  if (reason === "on_failure") return "backup"
  if (reason.startsWith("condition:")) return "tier-down"
  if (reason.startsWith("router:")) return "router"
  return reason
}

/**
 * The models a request to this policy would try, in order, and the ones left
 * out with the reason, compiled by the server without sending anything.
 *
 * A stored policy is sent as its spec, so the plan follows the copy on screen
 * rather than a server cache that may not have caught up with a save yet.
 */
export function PolicyTestGroup({
  row,
  workspaceId,
}: {
  row: RoutingRow
  workspaceId: string | null
}) {
  const [budgetUsed, setBudgetUsed] = useState("")
  const settledBudget = useDebounced(budgetUsed)
  const hasTierDowns = findBudgetConditions(row.spec).length > 0
  const budgetPct = Number(settledBudget)
  const isBudgetValid = settledBudget === "" || Number.isFinite(budgetPct)

  const explain = useExplainPolicy(
    {
      name: row.name,
      ...(row.source === "config" ? {} : { spec: row.spec }),
      ...(workspaceId === null ? {} : { workspace_id: workspaceId }),
      ...(hasTierDowns && settledBudget !== "" && isBudgetValid
        ? { budget_used_pct: budgetPct }
        : {}),
    },
    isBudgetValid,
  )
  const plan = explain.data

  return (
    <PolicyGroup title="Test" isReadOnly error={explain.error}>
      {hasTierDowns ? (
        <Field
          label="Budget used (%)"
          value={budgetUsed}
          onChange={setBudgetUsed}
          inputMode="decimal"
          isInvalid={!Number.isFinite(Number(budgetUsed))}
          errorMessage="Enter a number."
        />
      ) : null}
      {plan === undefined ? null : (
        <div className="flex flex-col gap-3">
          {plan.candidates.length === 0 ? (
            <p className="text-sm text-danger">
              No model can serve this policy.
            </p>
          ) : (
            <ol
              aria-label="Tried in order"
              className="flex list-decimal flex-col gap-1 pl-5 text-sm text-foreground"
            >
              {plan.candidates.map((candidate) => (
                <li key={candidate.position}>
                  {selectorText(`${candidate.instance}:${candidate.model}`)}
                  <span className="text-muted">
                    {" "}
                    · {reasonLabel(candidate.selection_reason)}
                  </span>
                </li>
              ))}
            </ol>
          )}
          {plan.dropped.length === 0 ? null : (
            <div className="flex flex-col gap-1">
              <p id="policy-test-dropped" className="text-caption">
                Left out
              </p>
              <ul
                aria-labelledby="policy-test-dropped"
                className="flex flex-col gap-1 text-sm text-muted"
              >
                {plan.dropped.map((item, index) => (
                  <li key={`${index}:${item.selector}`}>
                    {selectorText(item.selector)} · {item.detail}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </PolicyGroup>
  )
}
