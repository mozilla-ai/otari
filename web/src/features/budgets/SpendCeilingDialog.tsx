import { AlertDialog, Button } from "@heroui/react"
import { useEffect, useState } from "react"

import type {
  OrganizationBudget,
  OrganizationSpendCeiling,
  Workspace,
} from "@/client"
import { Field } from "@/shared/components/Field"
import { ErrorBanner, FilterSelect } from "@/shared/components/ui"

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
  isPending: boolean
  error: unknown
  onSubmit: (draft: SpendCeilingDraft) => void
}

export function SpendCeilingDialog({
  isOpen,
  onOpenChange,
  editing,
  budgets,
  workspaces,
  organizationName,
  isPending,
  error,
  onSubmit,
}: SpendCeilingDialogProps) {
  // "organization", or a workspace id. One control rather than a kind and an id,
  // because the two scopes this page creates are a closed list and asking for a
  // kind first would be a step with one real choice in it.
  const [target, setTarget] = useState(ORGANIZATION_SCOPE)
  const [budgetId, setBudgetId] = useState("")
  const [provider, setProvider] = useState("")
  const [name, setName] = useState("")

  useEffect(() => {
    if (!isOpen) return
    setTarget(ORGANIZATION_SCOPE)
    setBudgetId(editing?.budget_id ?? budgets[0]?.budget_id ?? "")
    setProvider(editing?.provider_key_id ?? "")
    setName(editing?.name ?? "")
  }, [isOpen, editing, budgets])

  const ownOptions = budgets.map((budget) => ({
    value: budget.budget_id,
    label: `${budgetLabel(budget)} — ${limitLabel(budget.max_budget)}`,
  }))
  // A ceiling holding a budget set at the deployment level opens on an id no
  // option carries, and `FilterSelect` renders such a value as itself: a raw
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
          label: `Set at the deployment level — ${limitLabel(editing.max_budget)}`,
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
  const blockedReason = noBudgets
    ? "Add a budget first. A ceiling enforces a budget, so there is nothing for this one to hold."
    : budgetId === ""
      ? "Choose the budget this ceiling enforces."
      : budgetIsOwned
        ? undefined
        : "This ceiling holds a budget set at the deployment level. Choose one of your own to take it over."

  const submit = () => {
    if (blockedReason !== undefined) return
    onSubmit({
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
    <AlertDialog isOpen={isOpen} onOpenChange={onOpenChange}>
      {isOpen ? (
        <AlertDialog.Backdrop>
          <AlertDialog.Container placement="center" size="md">
            <AlertDialog.Dialog>
              <AlertDialog.Header>
                <AlertDialog.Heading>
                  {editing ? "Edit spend ceiling" : "Add spend ceiling"}
                </AlertDialog.Heading>
              </AlertDialog.Header>
              <AlertDialog.Body className="flex flex-col gap-4">
                <p className="text-sm text-muted">
                  A ceiling holds one identity to one budget. Every ceiling that
                  applies to a request has to pass, so an organization-wide cap
                  and a workspace cap both bind.
                </p>
                <ErrorBanner error={error} />
                {editing ? (
                  <div className="flex flex-col gap-1">
                    <span className="text-sm font-medium text-foreground">
                      Capping
                    </span>
                    <span className="text-sm text-muted">
                      {scopeLabel(editing, { organizationName, workspaces })}
                      {editing.provider_key_id
                        ? `, on ${editing.provider_key_id}`
                        : ", on every provider"}
                    </span>
                    <span className="text-xs text-muted">
                      What a ceiling caps cannot be changed. Delete it and add
                      one for the other identity.
                    </span>
                  </div>
                ) : (
                  <>
                    <FilterSelect
                      label="Capping"
                      value={target}
                      onChange={setTarget}
                      options={targetOptions}
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
                <FilterSelect
                  label="Budget"
                  value={budgetId}
                  onChange={setBudgetId}
                  options={budgetOptions}
                  disabled={noBudgets}
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
                    This ceiling currently holds a budget set at the deployment
                    level. Choosing one of your own moves it, and leaves that
                    budget as it is.
                  </p>
                ) : null}
                {blockedReason ? (
                  <p className="text-sm text-warning">{blockedReason}</p>
                ) : null}
              </AlertDialog.Body>
              <AlertDialog.Footer>
                <Button
                  variant="ghost"
                  isDisabled={isPending}
                  onPress={() => onOpenChange(false)}
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  isDisabled={blockedReason !== undefined}
                  isPending={isPending}
                  onPress={submit}
                >
                  {editing ? "Save ceiling" : "Add ceiling"}
                </Button>
              </AlertDialog.Footer>
            </AlertDialog.Dialog>
          </AlertDialog.Container>
        </AlertDialog.Backdrop>
      ) : null}
    </AlertDialog>
  )
}
