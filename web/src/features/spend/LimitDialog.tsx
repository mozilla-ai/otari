import { useState } from "react"
import type { WorkspaceBudgetDefault } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { perWords } from "@/features/spend/limits"

type Period = Pick<
  WorkspaceBudgetDefault,
  "reset_alignment" | "budget_duration_sec"
>

const PERIOD_NOUN: Record<string, string> = {
  "an hour": "hour",
  "a day": "day",
  "a week": "week",
  "a month": "month",
}

/** The field's label and the line under it, for the period the limit runs on. */
function periodCopy(period: Period | undefined): {
  label: string
  description: string
} {
  if (!period || period.reset_alignment === "calendar_month") {
    return {
      label: "Monthly spend limit, in dollars",
      description:
        "Resets on the first of each month. A request that would go past it is refused, and the refusal shows in Sessions.",
    }
  }
  const per = perWords(period)
  const noun = PERIOD_NOUN[per]
  return {
    label: "Spend limit, in dollars",
    description: !per
      ? "A request that would go past it is refused, and the refusal shows in Sessions."
      : `A request that would take a person past the limit is refused until ${
          noun ? `their ${noun} resets` : "it resets"
        }, and the refusal shows in Sessions.`,
  }
}

/**
 * One amount: the whole of setting a workspace's limit for each person. A new
 * one resets each calendar month; an existing one keeps its own period, which
 * `period` names. otari stores it as a budget plus the default that points it
 * at the workspace; the caller makes both, so nobody here has to know that.
 */
export function LimitDialog({
  isOpen,
  onOpenChange,
  title,
  description,
  initial,
  period,
  isPending,
  error,
  onSave,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: string
  /** The limit already set, in dollars, or "" for none. */
  initial: string
  /** How often the existing limit resets; left out, a calendar month. */
  period?: Period
  isPending: boolean
  error: unknown
  onSave: (amount: number) => void
}) {
  const [value, setValue] = useState(initial)
  const amount = Number(value)
  const isValid = value.trim() !== "" && Number.isFinite(amount) && amount > 0
  const copy = periodCopy(period)
  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (open) setValue(initial)
        onOpenChange(open)
      }}
      title={title}
      description={description}
      submitLabel="Save limit"
      isPending={isPending}
      error={error}
      isDirty={value !== initial}
      isSubmitDisabled={!isValid}
      onSubmit={() => onSave(amount)}
    >
      <Field
        label={copy.label}
        value={value}
        onChange={setValue}
        inputMode="decimal"
        placeholder="50"
        description={copy.description}
        isInvalid={value.trim() !== "" && !isValid}
        errorMessage="Enter a number above zero, like 12.50."
        autoFocus
      />
    </FormDialog>
  )
}
