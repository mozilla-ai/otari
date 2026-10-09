import { type RefObject, useState } from "react"

import type { OrganizationBudget } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  useCreateOrganizationBudget,
  useOrganizationBudgets,
  useUpdateOrganizationBudget,
} from "@/shared/api/budgets"
import { AppliedToPicker } from "./AppliedToPicker"
import {
  describeEntity,
  entityFromKey,
  entityGroups,
  entityKey,
  takenEntities,
} from "./appliedEntities"
import { budgetLabeler, unnamedBudgetLabel } from "./budgetLabel"
import { ResetCycleField } from "./ResetCycleField"
import {
  type CycleDraft,
  type CycleFields,
  cycleDraftFrom,
  cycleFieldsFromDraft,
  findCycleProblem,
} from "./resetCycle"
import { useEntitySources } from "./useEntitySources"

// The form behind both Create and Edit for one of the organization's budgets: the
// limit, the reset cycle and the entities it applies to, saved as one write. One
// component rather than two: the fields are identical, and the endpoint is a
// PATCH that leaves an omitted field alone, so an edit sends the same shape an
// add does.

export interface OrganizationBudgetDraft extends CycleFields {
  name: string | null
  max_budget: number | null
}

/** A typed amount, or undefined when it is not a number this can send. */
function parseLimit(raw: string): number | undefined {
  const trimmed = raw.trim()
  // Blank is a deliberate value here, not a missing one: it means no limit.
  if (trimmed === "") return undefined
  const parsed = Number(trimmed)
  if (!Number.isFinite(parsed) || parsed < 0) return undefined
  return parsed
}

function limitToInput(value: number | null | undefined): string {
  return value === null || value === undefined ? "" : String(value)
}

export interface OrganizationBudgetDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** The budget being edited; absent means this is an add. */
  editing?: OrganizationBudget
  organizationId: string
  organizationName: string
  /** Called once a save has landed, so the caller can close this. */
  onSaved: () => void
  /** Where focus returns when the control that opened this is gone. */
  returnFocusRef?: RefObject<HTMLElement | null>
}

export function OrganizationBudgetDialog({
  isOpen,
  onOpenChange,
  editing,
  organizationId,
  organizationName,
  onSaved,
  returnFocusRef,
}: OrganizationBudgetDialogProps) {
  // The mutations live here, below the caller's key, so a refused save is
  // cleared by the same remount that clears the draft. See feedback.md, "The
  // component that renders the FormDialog owns everything that resets between
  // opens: the draft *and* its mutation".
  const create = useCreateOrganizationBudget()
  const update = useUpdateOrganizationBudget()

  const budgets = useOrganizationBudgets()
  // Read only while open: the page keeps this mounted between opens, and these
  // would otherwise load on every visit to the budgets page.
  const { sources, failedLists, isSettled } = useEntitySources(
    organizationId,
    isOpen,
  )
  // The server names the workspaces a budget applies to, so one still reads by
  // name while the workspace list is loading or failed to load.
  const namedWorkspaces = (editing?.applied_to ?? []).flatMap((entity) =>
    entity.scope_type === "workspace" && entity.name
      ? [{ id: entity.scope_id, name: entity.name }]
      : [],
  )

  // Seeded on mount only, because the caller remounts this on each open: these
  // values decide what colleagues may spend, so inheriting the last budget's
  // figure into a different one is the expensive kind of mistake. The entities
  // come with the budget, so an edit opens on the whole set it will send back.
  const [name, setName] = useState(editing?.name ?? "")
  const [limit, setLimit] = useState(limitToInput(editing?.max_budget))
  const [cycle, setCycle] = useState<CycleDraft>(cycleDraftFrom(editing))
  const [applied, setApplied] = useState<string[]>(() =>
    (editing?.applied_to ?? []).map(entityKey),
  )

  const amount = parseLimit(limit)
  const limitInvalid = limit.trim() !== "" && amount === undefined
  const { isDirty } = useDirtySnapshot({
    name,
    limit,
    cycle,
    applied: [...applied].sort(),
  })

  const nameBudget = budgetLabeler(budgets.data ?? [])
  const taken = takenEntities(
    budgets.data ?? [],
    editing?.budget_id,
    nameBudget,
  )
  const groups = entityGroups(sources, taken)
  const organizationKey = entityKey({
    scope_type: "organization",
    scope_id: organizationId,
  })

  // What this budget is shown as while it has no name of its own. The token and
  // request caps come from the budget being edited: this form does not offer
  // them, and a label derived without them would understate what it caps.
  const cycleWire = cycleFieldsFromDraft(cycle)
  const unnamedLabel = unnamedBudgetLabel({
    max_budget: amount ?? null,
    token_limit: editing?.token_limit ?? null,
    request_limit: editing?.request_limit ?? null,
    ...cycleWire,
  })

  const submit = () => {
    if (limitInvalid || findCycleProblem(cycle) !== undefined) return
    // The whole cycle set every time, never one field: the server refuses a
    // budget holding a setting its cycle does not take.
    const body = {
      name: name.trim() === "" ? null : name.trim(),
      max_budget: amount ?? null,
      ...cycleWire,
      applied_to: applied.map(entityFromKey),
    }
    const onDone = { onSuccess: onSaved }
    if (editing) update.mutate({ id: editing.budget_id, body }, onDone)
    else create.mutate(body, onDone)
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="xl"
      title={editing ? "Edit budget" : "New budget"}
      description="A budget is a limit, a reset cycle, and the entities it applies to."
      submitLabel={editing ? "Save budget" : "Create budget"}
      onSubmit={submit}
      isPending={create.isPending || update.isPending}
      isSubmitDisabled={limitInvalid}
      isDirty={isDirty}
      error={editing ? update.error : create.error}
      returnFocusRef={returnFocusRef}
    >
      <Field
        label="Name"
        value={name}
        onChange={setName}
        placeholder="Engineering monthly"
        autoFocus
        description={`Optional. Left blank, this budget is called "${unnamedLabel}".`}
      />
      <Field
        label="Limit (USD)"
        value={limit}
        onChange={setLimit}
        placeholder="250"
        isInvalid={limitInvalid}
        errorMessage="Enter an amount of zero or more, or leave it blank for no limit."
        // "no dollar limit", not "admits every request": a budget capping
        // tokens or requests still refuses.
        description="Leave blank for no dollar limit."
      />
      <ResetCycleField value={cycle} onChange={setCycle} />
      {failedLists.length > 0 ? (
        <InfoBanner tone="warning">
          Could not load {failedLists.join(", ")}, so those cannot be picked.
          Entities this budget already applies to are kept. Close this and try
          again to pick them.
        </InfoBanner>
      ) : null}
      <AppliedToPicker
        value={applied}
        onChange={setApplied}
        groups={groups}
        organizationKey={organizationKey}
        organizationName={organizationName}
        organizationTaken={taken.get(organizationKey)}
        isLoading={!isSettled}
        describe={(key) =>
          describeEntity(entityFromKey(key), {
            organizationName,
            workspaces: [...sources.workspaces, ...namedWorkspaces],
          })
        }
      />
      {editing && editing.ceiling_count > 0 ? (
        <p className="text-sm text-muted">
          Spend already recorded stays; a new limit or cycle applies from here
          on.
        </p>
      ) : null}
    </FormDialog>
  )
}
