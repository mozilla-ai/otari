import { useState } from "react"

import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { useKeys } from "@/shared/api/apiKeys"
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
  if (reason.startsWith("router:")) {
    const name = reason.slice("router:".length)
    return name ? `router ${name}` : "router"
  }
  return reason
}

/**
 * The models a request to this policy would try, in order, and the ones left
 * out with the reason, compiled by the server without sending anything.
 *
 * A stored policy is sent as its spec, so the plan follows the copy on screen
 * rather than a server cache that may not have caught up with a save yet.
 *
 * A key can be named to compile as that caller: its allow-list is what decides
 * whether the plan compiles down, which is the failure this section exists to
 * catch before it happens in production.
 */
export function PolicyTestGroup({
  row,
  workspaceId,
}: {
  row: RoutingRow
  workspaceId: string | null
}) {
  const [budgetUsed, setBudgetUsed] = useState("")
  const [keyId, setKeyId] = useState("")
  // Trimmed, since `Number(" ")` is 0 and would read as a real 0%.
  const settledBudget = useDebounced(budgetUsed.trim())
  const hasTierDowns = findBudgetConditions(row.spec).length > 0
  const budgetPct = Number(settledBudget)
  const isBudgetValid = settledBudget === "" || Number.isFinite(budgetPct)

  const { data: keys } = useKeys(workspaceId ?? undefined)
  const key = keys?.find((candidate) => candidate.id === keyId) ?? null
  const keyOptions = [
    { value: "", label: "Any caller (unrestricted)" },
    ...(keys ?? []).map((candidate) => ({
      value: candidate.id,
      // `key_name` is optional on a key, and the prefix is what the keys page
      // falls back to, so a picker over unnamed keys is still usable.
      label: candidate.key_name ?? candidate.key_prefix ?? candidate.id,
      description:
        candidate.allowed_models === null
          ? "May use any model."
          : candidate.allowed_models.length === 0
            ? "May use no model."
            : `May use ${candidate.allowed_models.join(", ")}.`,
    })),
  ]

  const explain = useExplainPolicy(
    {
      name: row.name,
      ...(row.source === "config" ? {} : { spec: row.spec }),
      ...(workspaceId === null ? {} : { workspace_id: workspaceId }),
      // `null` on a key means unrestricted, which is the same as sending none.
      ...(key === null
        ? {}
        : {
            key_id: key.id,
            ...(key.allowed_models === null
              ? {}
              : { allowed_models: key.allowed_models }),
          }),
      ...(hasTierDowns && settledBudget !== "" && isBudgetValid
        ? { budget_used_pct: budgetPct }
        : {}),
    },
    isBudgetValid,
  )
  const isFieldValid = Number.isFinite(Number(budgetUsed.trim()))
  // An invalid budget compiles nothing, so the last plan would answer a value
  // that is no longer in the field.
  const plan = isFieldValid ? explain.data : undefined
  // While a new value settles and compiles, the last plan stays in place so the
  // section does not jump, dimmed so it does not read as the answer.
  const isStale =
    explain.isPlaceholderData || budgetUsed.trim() !== settledBudget
  // A weighted split is written in the policy, so it is knowable without a
  // request. Keyed by the selector the policy declares, which is why it is a
  // line of its own rather than a share beside each resolved candidate.
  const split = Object.entries(plan?.router_weights ?? {}).map(
    ([selector, share]) => `${selectorText(selector)} ${Math.round(share)}%`,
  )
  // A plan headed by the default target while the policy declares a router means
  // the router did not supply an ordering, which is a backend this build lacks.
  const isRouterMissing =
    plan !== undefined &&
    plan.router_backend !== null &&
    plan.candidates.length > 0 &&
    plan.selection_reason === "default"

  return (
    <PolicyGroup title="Test" isReadOnly error={explain.error}>
      <Select
        label="As API key"
        value={keyId}
        onChange={setKeyId}
        options={keyOptions}
        description="Compile as a caller holding this key, so its allow-list decides what the policy can reach."
      />
      {hasTierDowns ? (
        <Field
          label="Budget used (%)"
          value={budgetUsed}
          onChange={setBudgetUsed}
          inputMode="decimal"
          isInvalid={!isFieldValid}
          errorMessage="Enter a number."
        />
      ) : null}
      {plan === undefined ? null : (
        <div
          aria-busy={isStale}
          className={`flex flex-col gap-3${isStale ? " opacity-50" : ""}`}
        >
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
          {split.length === 0 ? null : (
            <p className="text-caption">Traffic split: {split.join(", ")}.</p>
          )}
          {isRouterMissing ? (
            <p className="text-caption">
              This build has no router named {plan.router_backend}, so the plan
              above is the policy&apos;s default target.
            </p>
          ) : null}
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
