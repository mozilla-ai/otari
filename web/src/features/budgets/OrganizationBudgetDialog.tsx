import { type RefObject, useState } from "react"

import type { OrganizationBudget } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  useCreateOrganizationBudget,
  useUpdateOrganizationBudget,
} from "@/shared/api/budgets"

import { unnamedBudgetLabel } from "./budgetLabel"
import { ResetCycleField } from "./ResetCycleField"
import {
  type CycleDraft,
  type CycleFields,
  cycleDraftFrom,
  cycleFieldsFromDraft,
  findCycleProblem,
} from "./resetCycle"

// The form behind both Create and Edit for one of the organization's budgets. One
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
  /** Called once a save has landed, so the caller can close this. */
  onSaved: () => void
  /** Where focus returns when the control that opened this is gone. */
  returnFocusRef?: RefObject<HTMLElement | null>
}

export function OrganizationBudgetDialog({
  isOpen,
  onOpenChange,
  editing,
  onSaved,
  returnFocusRef,
}: OrganizationBudgetDialogProps) {
  // The mutations live here, below the caller's key, so a refused save is
  // cleared by the same remount that clears the draft. See feedback.md, "The
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
    cycle: cycleDraftFrom(editing),
  }
  const [name, setName] = useState(seed.name)
  const [limit, setLimit] = useState(seed.limit)
  const [cycle, setCycle] = useState<CycleDraft>(seed.cycle)

  const amount = parseLimit(limit)
  const limitInvalid = limit.trim() !== "" && amount === undefined
  // The whole draft against what it was seeded with, so what "unsaved" means
  // cannot drift from what the form holds.
  const { isDirty } = useDirtySnapshot({ name, limit, cycle })

  // What this budget is shown as while it has no name of its own, said on the
  // field so the admin sees the label before saving rather than after. On the
  // description rather than as the placeholder, which design/forms.md reserves
  // for an example of what to type. The token and request caps come from the
  // budget being edited: this form does not offer them, and a label derived
  // without them would understate what it caps.
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
    // budget holding a setting its cycle does not take, so a weekday mask left
    // behind by a switch to Monthly is what makes an otherwise valid save fail.
    save({
      name: name.trim() === "" ? null : name.trim(),
      max_budget: amount ?? null,
      ...cycleWire,
    })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      title={editing ? "Edit budget" : "New budget"}
      description="A budget is an amount, a reset cycle, and the entities it applies to. It caps nothing on its own: a spend ceiling is what points it at an organization, a workspace, or a key."
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
        description={`Optional. What this budget is called wherever it is handed out. Left blank, that is "${unnamedLabel}".`}
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
      <ResetCycleField value={cycle} onChange={setCycle} />
      {editing && editing.ceiling_count > 0 ? (
        <p className="text-sm text-muted">
          {editing.ceiling_count === 1
            ? "1 spend ceiling is held to this budget and moves with it."
            : `${editing.ceiling_count} spend ceilings are held to this budget and move with it.`}{" "}
          Spend already recorded stays; the new figure applies from here on.
        </p>
      ) : null}
    </FormDialog>
  )
}
