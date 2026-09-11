import { useState } from "react"

import type { OrganizationBudget } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  useCreateOrganizationBudget,
  useUpdateOrganizationBudget,
} from "@/shared/api/budgets"

import {
  PERIOD_OPTIONS,
  periodValue,
  type ResetAlignment,
} from "./organizationBudget"

// The form behind both Add and Edit for one of the organization's budgets. One
// component rather than two: the fields are identical, and the endpoint is a
// PATCH that leaves an omitted field alone, so an edit sends the same shape an
// add does.

export interface OrganizationBudgetDraft {
  name: string | null
  max_budget: number | null
  budget_duration_sec: number | null
  // The generated union, not `string`: see `ResetAlignment` for why restating it
  // as a string broke the build when the endpoint narrowed the field.
  reset_alignment: ResetAlignment | null
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
  /** Called once a save has landed, so the caller can close this. */
  onSaved: () => void
}

export function OrganizationBudgetDialog({
  isOpen,
  onOpenChange,
  editing,
  onSaved,
}: OrganizationBudgetDialogProps) {
  // The mutations live here, below the caller's key, so a refused save is
  // cleared by the same remount that clears the draft. Held in the card they
  // outlived it: a refusal's banner greeted the next open, and a failed edit of
  // one row was what the next row's dialog showed. See feedback.md, "The
  // component that renders the FormDialog owns everything that resets between
  // opens: the draft *and* its mutation".
  const create = useCreateOrganizationBudget()
  const update = useUpdateOrganizationBudget()
  const save = (draft: OrganizationBudgetDraft) => {
    const onDone = { onSuccess: onSaved }
    if (editing) {
      update.mutate({ id: editing.budget_id, body: draft }, onDone)
      return
    }
    create.mutate(draft, onDone)
  }
  // Seeded on mount only, because the caller remounts this on each open. That
  // matters more here than on most forms: these values decide what colleagues
  // may spend, so inheriting the last budget's figure into a different one is
  // the expensive kind of mistake.
  const seed = {
    name: editing?.name ?? "",
    limit: limitToInput(editing?.max_budget),
    period: periodValue(editing),
  }
  const [name, setName] = useState(seed.name)
  const [limit, setLimit] = useState(seed.limit)
  const [period, setPeriod] = useState(seed.period)

  const amount = parseLimit(limit)
  const limitInvalid = limit.trim() !== "" && amount === undefined
  // The whole draft against what it was seeded with, so what "unsaved" means
  // cannot drift from what the form holds.
  const { isDirty } = useDirtySnapshot({ name, limit, period })

  // A duration-carrying budget (one the deployment surface created) opens on
  // "No reset", and saving would clear the duration rather than keep it. Said
  // out loud, because the form cannot show a period it does not offer.
  const clearsDuration =
    editing !== undefined &&
    !editing.reset_alignment &&
    editing.budget_duration_sec !== null &&
    editing.budget_duration_sec !== undefined &&
    period === "none"

  const submit = () => {
    if (limitInvalid) return
    const option = PERIOD_OPTIONS.find(
      (candidate) => candidate.value === period,
    )
    save({
      name: name.trim() === "" ? null : name.trim(),
      max_budget: amount ?? null,
      // Only ever one of the two is sent with a value, because a budget resets
      // on a duration or on a boundary and the database refuses both.
      budget_duration_sec: null,
      reset_alignment: option?.alignment ?? null,
    })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      title={editing ? "Edit budget" : "New budget"}
      description="A budget is an amount and the period it is spent over. It caps nothing on its own: a spend ceiling is what points it at an organization, a workspace, or a key."
      submitLabel={editing ? "Save budget" : "Add budget"}
      onSubmit={submit}
      isPending={create.isPending || update.isPending}
      isSubmitDisabled={limitInvalid}
      isDirty={isDirty}
      error={editing ? update.error : create.error}
    >
      <Field
        label="Name"
        value={name}
        onChange={setName}
        placeholder="Engineering monthly"
        autoFocus
        description="Optional. What this budget is called wherever it is handed out."
      />
      <Field
        label="Limit (USD)"
        value={limit}
        onChange={setLimit}
        placeholder="250"
        isInvalid={limitInvalid}
        errorMessage="Enter an amount of zero or more, or leave it blank for no limit."
        // "no dollar limit", not "admits every request": a budget capping
        // tokens or requests still refuses, so the old wording is a claim about
        // behavior this field no longer decides alone.
        description="Leave blank for no dollar limit."
      />
      <Select
        label="Resets"
        value={period}
        onChange={setPeriod}
        options={PERIOD_OPTIONS.map((option) => ({
          value: option.value,
          label: option.label,
        }))}
        reserveMessage={false}
      />
      {editing && editing.ceiling_count > 0 ? (
        <p className="text-sm text-muted">
          {editing.ceiling_count === 1
            ? "1 spend ceiling is held to this budget and moves with it."
            : `${editing.ceiling_count} spend ceilings are held to this budget and move with it.`}{" "}
          Spend already recorded stays; the new figure applies from here on.
        </p>
      ) : null}
      {clearsDuration ? (
        <p className="text-sm text-warning">
          This budget currently resets on a rolling interval, which this form
          does not offer. Saving replaces it with the period chosen above.
        </p>
      ) : null}
    </FormDialog>
  )
}
