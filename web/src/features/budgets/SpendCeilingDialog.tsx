import { useState } from "react"

import type {
  OrganizationBudget,
  OrganizationSpendCeiling,
  Workspace,
} from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  useCreateOrganizationSpendCeiling,
  useUpdateOrganizationSpendCeiling,
} from "@/shared/api/budgets"

import { budgetLabel, limitLabel, scopeLabel } from "./organizationBudget"

// The form behind both Add and Edit for a spend ceiling.
//
// Add and Edit differ here in a way the budget dialog's do not, and the
// difference is the endpoint's: the scope and the provider narrowing are not
// editable, because changing either would move the ceiling to a different
// identity while carrying its spend, which is a delete and a create. So an edit
// shows them as read-only facts and offers only the label and the budget.

export interface SpendCeilingDraft {
  scope_type: "organization" | "workspace"
  scope_id: string
  provider_key_id: string | null
  budget_id: string
  name: string | null
}

const ORGANIZATION_SCOPE = "organization"

export interface SpendCeilingDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** The ceiling being edited; absent means this is an add. */
  editing?: OrganizationSpendCeiling
  /** The organization's own budgets, which are the only ones a ceiling may name. */
  budgets: readonly OrganizationBudget[]
  workspaces: readonly Workspace[]
  organizationName: string
  /** Called once a save has landed, so the caller can close this. */
  onSaved: () => void
}

export function SpendCeilingDialog({
  isOpen,
  onOpenChange,
  editing,
  budgets,
  workspaces,
  organizationName,
  onSaved,
}: SpendCeilingDialogProps) {
  // Below the caller's key with the draft, so a refused save cannot greet the
  // next open (feedback.md: the component that renders the FormDialog owns the
  // draft *and* its mutation).
  const create = useCreateOrganizationSpendCeiling()
  const update = useUpdateOrganizationSpendCeiling()
  const save = (draft: SpendCeilingDraft) => {
    const onDone = { onSuccess: onSaved }
    if (editing) {
      // Only the two fields the endpoint accepts on a PATCH. Sending the scope
      // would be ignored, and sending it anyway would suggest it could change.
      update.mutate(
        {
          id: editing.id,
          body: { budget_id: draft.budget_id, name: draft.name },
        },
        onDone,
      )
      return
    }
    create.mutate(draft, onDone)
  }
  // "organization", or a workspace id. One control rather than a kind and an id,
  // because the two scopes this page creates are a closed list and asking for a
  // kind first would be a step with one real choice in it.
  // Seeded on mount only, because the caller remounts this on each open.
  const seed = {
    target: ORGANIZATION_SCOPE,
    budgetId: editing?.budget_id ?? budgets[0]?.budget_id ?? "",
    provider: editing?.provider_key_id ?? "",
    name: editing?.name ?? "",
  }
  const [target, setTarget] = useState(seed.target)
  const [budgetId, setBudgetId] = useState(seed.budgetId)
  const [provider, setProvider] = useState(seed.provider)
  const [name, setName] = useState(seed.name)
  // The whole draft against what it was seeded with, so what "unsaved" means
  // cannot drift from what the form holds.
  const { isDirty, reset: reseed } = useDirtySnapshot({
    target,
    budgetId,
    provider,
    name,
  })
  // The default budget is part of the seed and can arrive after mount: the
  // opener does not wait for the list, so a dialog opened first seeds `""`.
  // Applied here, and seeded with the explicit value, because this render still
  // holds the empty one; a `seed` recomputed per render would instead make the
  // guard track the props and ask to discard a form nobody typed in.
  const [budgetSeeded, setBudgetSeeded] = useState(seed.budgetId !== "")
  if (!budgetSeeded && editing === undefined && budgets.length > 0) {
    const landed = budgets[0].budget_id
    setBudgetSeeded(true)
    setBudgetId(landed)
    reseed({ target, budgetId: landed, provider, name })
  }

  const ownOptions = budgets.map((budget) => ({
    value: budget.budget_id,
    label: `${budgetLabel(budget)} — ${limitLabel(budget)}`,
  }))
  // A ceiling holding a budget set at the deployment level opens on an id no
  // option carries, and `Select` renders such a value as itself: a raw
  // uuid where the budget's name belongs. Carried as its own labelled option
  // instead, so the current selection reads as what it is and choosing one of
  // the organization's own is still the way out.
  const editingIsForeign =
    editing !== undefined &&
    !editing.manageable &&
    !budgets.some((budget) => budget.budget_id === editing.budget_id)
  const budgetOptions = editingIsForeign
    ? [
        {
          value: editing.budget_id,
          label: `Set at the deployment level — ${limitLabel(editing)}`,
        },
        ...ownOptions,
      ]
    : ownOptions

  const targetOptions = [
    {
      value: ORGANIZATION_SCOPE,
      label: `${organizationName} (whole organization)`,
    },
    ...workspaces.map((workspace) => ({
      value: workspace.id,
      label: `${workspace.name} (workspace)`,
    })),
  ]

  const noBudgets = budgetOptions.length === 0
  // Labelling the deployment-level option is what makes the current selection
  // readable; this is what stops it being *saved* unchanged. The endpoint
  // resolves a ceiling's budget against the organization's own, so submitting
  // that id is a guaranteed 404, and an enabled Save that always fails is worse
  // than one that says what it needs.
  const budgetIsOwned = budgets.some((budget) => budget.budget_id === budgetId)
  // Two of these are about the Budget control, so they are announced on it
  // rather than as a paragraph five controls below the choice they describe.
  // `noBudgets` is about the list rather than the choice, so it stays prose.
  const budgetReason = noBudgets
    ? undefined
    : budgetId === ""
      ? "Choose the budget this ceiling enforces."
      : budgetIsOwned
        ? undefined
        : "This ceiling holds a budget set at the deployment level. Choose one of your own to take it over."
  const listReason = noBudgets
    ? "Add a budget first. A ceiling enforces a budget, so there is nothing for this one to hold."
    : undefined
  const blockedReason = listReason ?? budgetReason

  const submit = () => {
    if (blockedReason !== undefined) return
    save({
      scope_type: target === ORGANIZATION_SCOPE ? "organization" : "workspace",
      // The editing path never reaches here with a changed scope: the endpoint
      // ignores both fields on a PATCH and the controls are not rendered.
      scope_id: editing?.scope_id ?? target,
      provider_key_id: provider.trim() === "" ? null : provider.trim(),
      budget_id: budgetId,
      name: name.trim() === "" ? null : name.trim(),
    })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      title={editing ? "Edit spend ceiling" : "New spend ceiling"}
      description="A ceiling holds one identity to one budget. Every ceiling that applies to a request has to pass, so an organization-wide cap and a workspace cap both bind."
      submitLabel={editing ? "Save ceiling" : "Add ceiling"}
      onSubmit={submit}
      isPending={create.isPending || update.isPending}
      isSubmitDisabled={blockedReason !== undefined}
      isDirty={isDirty}
      error={editing ? update.error : create.error}
    >
      {editing ? (
        <div className="flex flex-col gap-1">
          <span className="text-body">Capping</span>
          <span className="text-sm text-muted">
            {scopeLabel(editing, { organizationName, workspaces })}
            {editing.provider_key_id
              ? `, on ${editing.provider_key_id}`
              : ", on every provider"}
          </span>
          <span className="text-caption">
            What a ceiling caps cannot be changed. Delete it and add one for the
            other identity.
          </span>
        </div>
      ) : (
        <>
          <Select
            label="Capping"
            value={target}
            onChange={setTarget}
            options={targetOptions}
            reserveMessage={false}
          />
          <Field
            label="Provider instance"
            value={provider}
            onChange={setProvider}
            placeholder="openai-eu"
            description="Optional. Narrows the cap to one provider; leave blank to cap spend across every provider."
          />
        </>
      )}
      <Select
        label="Budget"
        value={budgetId}
        onChange={setBudgetId}
        options={budgetOptions}
        isDisabled={noBudgets}
        isInvalid={budgetReason !== undefined}
        errorMessage={budgetReason}
        // Reserved, so announcing a refusal here does not move the footer.
        reserveMessage
      />
      <Field
        label="Name"
        value={name}
        onChange={setName}
        placeholder="Whole organization"
        description="Optional. A label for this ceiling, separate from the budget's own name."
      />
      {editing && !editing.manageable ? (
        <p className="text-sm text-muted">
          This ceiling currently holds a budget set at the deployment level.
          Choosing one of your own moves it, and leaves that budget as it is.
        </p>
      ) : null}
      {listReason ? <p className="text-sm text-warning">{listReason}</p> : null}
    </FormDialog>
  )
}
