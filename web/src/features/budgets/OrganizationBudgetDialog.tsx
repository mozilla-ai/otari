import { AlertDialog, Button } from "@heroui/react"
import { useEffect, useState } from "react"

import type { OrganizationBudget } from "@/client"
import { Field } from "@/shared/components/Field"
import { ErrorBanner, FilterSelect } from "@/shared/components/ui"

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
  isPending: boolean
  error: unknown
  onSubmit: (draft: OrganizationBudgetDraft) => void
}

export function OrganizationBudgetDialog({
  isOpen,
  onOpenChange,
  editing,
  isPending,
  error,
  onSubmit,
}: OrganizationBudgetDialogProps) {
  const [name, setName] = useState("")
  const [limit, setLimit] = useState("")
  const [period, setPeriod] = useState("calendar_month")

  // Reseeded every time it opens, for the reason `PricingOverrideDialog` gives:
  // the dialog stays mounted across close and reopen, and these values decide
  // what colleagues may spend, so inheriting the last budget's figure into a
  // different one is the expensive kind of mistake.
  useEffect(() => {
    if (!isOpen) return
    setName(editing?.name ?? "")
    setLimit(limitToInput(editing?.max_budget))
    setPeriod(periodValue(editing))
  }, [isOpen, editing])

  const amount = parseLimit(limit)
  const limitInvalid = limit.trim() !== "" && amount === undefined

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
    onSubmit({
      name: name.trim() === "" ? null : name.trim(),
      max_budget: amount ?? null,
      // Only ever one of the two is sent with a value, because a budget resets
      // on a duration or on a boundary and the database refuses both.
      budget_duration_sec: null,
      reset_alignment: option?.alignment ?? null,
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
                  {editing ? "Edit budget" : "Add budget"}
                </AlertDialog.Heading>
              </AlertDialog.Header>
              <AlertDialog.Body className="flex flex-col gap-4">
                <p className="text-sm text-muted">
                  A budget is an amount and the period it is spent over. It caps
                  nothing on its own: a spend ceiling below is what points it at
                  an organization, a workspace, or a key.
                </p>
                <ErrorBanner error={error} />
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
                  // "no dollar limit", not "admits every request": a budget
                  // capping tokens or requests still refuses, so the old wording
                  // is a claim about behavior this field no longer decides alone.
                  description="Leave blank for no dollar limit."
                />
                <FilterSelect
                  label="Resets"
                  value={period}
                  onChange={setPeriod}
                  options={PERIOD_OPTIONS.map((option) => ({
                    value: option.value,
                    label: option.label,
                  }))}
                />
                {editing && editing.ceiling_count > 0 ? (
                  <p className="text-sm text-muted">
                    {editing.ceiling_count === 1
                      ? "1 spend ceiling is held to this budget and moves with it."
                      : `${editing.ceiling_count} spend ceilings are held to this budget and move with it.`}{" "}
                    Spend already recorded stays; the new figure applies from
                    here on.
                  </p>
                ) : null}
                {clearsDuration ? (
                  <p className="text-sm text-warning">
                    This budget currently resets on a rolling interval, which
                    this form does not offer. Saving replaces it with the period
                    chosen above.
                  </p>
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
                  isDisabled={limitInvalid}
                  isPending={isPending}
                  onPress={submit}
                >
                  {editing ? "Save budget" : "Add budget"}
                </Button>
              </AlertDialog.Footer>
            </AlertDialog.Dialog>
          </AlertDialog.Container>
        </AlertDialog.Backdrop>
      ) : null}
    </AlertDialog>
  )
}
