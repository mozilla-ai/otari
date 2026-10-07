import { useState } from "react"

import type { OrganizationBudget } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { useKeys } from "@/shared/api/apiKeys"
import {
  useCreateOrganizationBudget,
  useOrganizationBudgets,
  useOrganizationSpendCeilingsAll,
  useUpdateOrganizationBudget,
} from "@/shared/api/budgets"
import { useModels } from "@/shared/api/models"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { useWorkspaces } from "@/shared/api/workspaces"
import { AppliedToPicker } from "./AppliedToPicker"
import {
  ceilingKey,
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
  cycleDraftError,
  cycleDraftFrom,
  cycleFieldsFromDraft,
} from "./resetCycle"

// The form behind both New and Edit for one of the organization's budgets: the
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
}

export function OrganizationBudgetDialog({
  isOpen,
  onOpenChange,
  editing,
  organizationId,
  organizationName,
  onSaved,
}: OrganizationBudgetDialogProps) {
  // The mutations live here, below the caller's key, so a refused save is
  // cleared by the same remount that clears the draft. See feedback.md, "The
  // component that renders the FormDialog owns everything that resets between
  // opens: the draft *and* its mutation".
  const create = useCreateOrganizationBudget()
  const update = useUpdateOrganizationBudget()

  const budgets = useOrganizationBudgets()
  const ceilings = useOrganizationSpendCeilingsAll()
  const workspaces = useWorkspaces()
  const members = useOrganizationMembers()
  const keys = useKeys()
  const models = useModels()

  // What this budget already applies to, read from the ceilings naming it.
  const heldKeys = (data: typeof ceilings.data) =>
    editing
      ? (data ?? [])
          .filter((ceiling) => ceiling.budget_id === editing.budget_id)
          .map(ceilingKey)
      : []

  // Seeded on mount only, because the caller remounts this on each open: these
  // values decide what colleagues may spend, so inheriting the last budget's
  // figure into a different one is the expensive kind of mistake.
  const [name, setName] = useState(editing?.name ?? "")
  const [limit, setLimit] = useState(limitToInput(editing?.max_budget))
  const [cycle, setCycle] = useState<CycleDraft>(cycleDraftFrom(editing))
  const [applied, setApplied] = useState<string[]>(() =>
    heldKeys(ceilings.data),
  )

  const amount = parseLimit(limit)
  const limitInvalid = limit.trim() !== "" && amount === undefined
  const { isDirty, reset: reseed } = useDirtySnapshot({
    name,
    limit,
    cycle,
    applied: [...applied].sort(),
  })

  // An edit sends the whole entity set, so it may only be sent once the set it
  // started from has landed: a save before that would remove every entity the
  // form had not read yet. The held set is part of the seed rather than a
  // change, so it is applied here, during render, with the snapshot reseeded.
  const [appliedSeeded, setAppliedSeeded] = useState(
    editing === undefined || ceilings.data !== undefined,
  )
  if (!appliedSeeded && ceilings.data !== undefined) {
    const held = heldKeys(ceilings.data)
    setAppliedSeeded(true)
    setApplied(held)
    reseed({ name, limit, cycle, applied: [...held].sort() })
  }

  const nameBudget = budgetLabeler(budgets.data ?? [])
  const taken = takenEntities(ceilings.data ?? [], editing?.budget_id, (id) => {
    const budget = budgets.data?.find((row) => row.budget_id === id)
    return budget ? nameBudget(budget) : undefined
  })
  const groups = entityGroups(
    {
      organizationId,
      workspaces: workspaces.data ?? [],
      members: members.data ?? [],
      keys: keys.data ?? [],
      modelIds: (models.data?.data ?? []).map((model) => model.id),
    },
    taken,
  )
  const organizationKey = entityKey({
    scope_type: "organization",
    scope_id: organizationId,
  })

  // Named, so an empty list reads as a failed read rather than as nothing to pick.
  const failedLists = [
    workspaces.isError && "workspaces",
    members.isError && "members",
    keys.isError && "API keys",
    models.isError && "providers and models",
  ].filter(Boolean)

  const seedReason = appliedSeeded
    ? undefined
    : ceilings.isError
      ? "Where this budget applies could not be read, so saving could remove entities it has. Close this and try again."
      : "Reading where this budget applies…"

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
    if (
      limitInvalid ||
      seedReason !== undefined ||
      cycleDraftError(cycle) !== undefined
    )
      return
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
      isSubmitDisabled={limitInvalid || seedReason !== undefined}
      isDirty={isDirty}
      error={editing ? update.error : create.error}
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
      {seedReason ? (
        <p
          className={`text-sm ${ceilings.isError ? "text-warning" : "text-muted"}`}
        >
          {seedReason}
        </p>
      ) : (
        <AppliedToPicker
          value={applied}
          onChange={setApplied}
          groups={groups}
          organizationKey={organizationKey}
          organizationName={organizationName}
          organizationTaken={taken.get(organizationKey)}
          describe={(key) =>
            describeEntity(entityFromKey(key), {
              organizationName,
              workspaces: workspaces.data ?? [],
            })
          }
        />
      )}
      {failedLists.length > 0 ? (
        <p className="text-sm text-warning">
          Could not load {failedLists.join(", ")}. Entities already applied are
          kept.
        </p>
      ) : null}
      {editing && editing.ceiling_count > 0 ? (
        <p className="text-sm text-muted">
          Spend already recorded stays; a new limit or cycle applies from here
          on.
        </p>
      ) : null}
    </FormDialog>
  )
}
