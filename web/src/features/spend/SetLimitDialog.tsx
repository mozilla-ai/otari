import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { ConfirmButton } from "@/design-system/actions/ConfirmButton"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { Segmented } from "@/design-system/navigation/Segmented"
import {
  HOUR_S,
  LIMIT_PERIODS,
  type LimitPeriod,
  type LimitTarget,
  useLimitWriter,
} from "@/features/spend/limits"

const PRESETS = [25, 50, 100, 250]

/**
 * The period a limit already runs on, as one the form offers, or null for one
 * it cannot name: a rolling 10 days, 24 hours, or none at all.
 */
function periodOf(target: LimitTarget): LimitPeriod | null {
  const ceiling = target.own?.ceiling
  if (!ceiling) return "calendar_month"
  const alignment = ceiling.reset_alignment
  if (
    alignment === "calendar_day" ||
    alignment === "calendar_week" ||
    alignment === "calendar_month"
  ) {
    return alignment
  }
  return ceiling.budget_duration_sec === HOUR_S ? "hour" : null
}

/** How a limit the form cannot name resets, said plainly: "every 7 days". */
function currentPeriodWords(seconds: number | null | undefined): string {
  if (!seconds) return "It never resets now."
  const days = seconds / 86_400
  const every =
    days < 1
      ? `${Math.round(seconds / HOUR_S)} hours`
      : days === 1
        ? "24 hours"
        : `${Math.round(days)} days`
  return `It resets every ${every} now, which is not one of these.`
}

/**
 * One amount and how often it resets: the whole of a spend limit, on whoever
 * the row is. Opened from the row it limits, so nobody goes looking for a
 * budgets page to cap a person.
 */
export function SetLimitDialog({
  target,
  title,
  description,
  onClose,
}: {
  target: LimitTarget
  title: string
  description: string
  onClose: () => void
}) {
  const writer = useLimitWriter()
  const initialAmount = target.own ? String(target.own.max) : ""
  const initialPeriod = periodOf(target)
  const [value, setValue] = useState(initialAmount)
  const [period, setPeriod] = useState<LimitPeriod | null>(initialPeriod)
  const amount = Number(value)
  const isValid = value.trim() !== "" && Number.isFinite(amount) && amount > 0

  const save = async () => {
    // A changed period needs a budget of its own. Either way the ceiling
    // stays, so what has been spent this period still counts.
    const isNewPeriod =
      target.own !== null && period !== null && period !== initialPeriod
    const isDone = isNewPeriod
      ? await writer.replace(target, amount, period)
      : await writer.save(target, amount, period)
    if (isDone) onClose()
  }

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={title}
      description={description}
      submitLabel="Save limit"
      isPending={writer.isPending}
      error={writer.error}
      isDirty={value !== initialAmount || period !== initialPeriod}
      isSubmitDisabled={!isValid}
      onSubmit={() => void save()}
      footerStart={
        target.own ? (
          <ConfirmButton
            confirmLabel="Remove limit"
            isPending={writer.isPending}
            onConfirm={() =>
              void writer.remove(target).then((isDone) => {
                if (isDone) onClose()
              })
            }
          >
            Remove limit
          </ConfirmButton>
        ) : undefined
      }
    >
      <div className="flex flex-col gap-2">
        <Field
          label="Spend limit, in dollars"
          value={value}
          onChange={setValue}
          inputMode="decimal"
          placeholder="50"
          isInvalid={value.trim() !== "" && !isValid}
          errorMessage="Enter a number above zero, like 12.50."
          shouldReserveMessage={false}
          autoFocus
        />
        <div className="otari-actions flex flex-wrap gap-1">
          {PRESETS.map((preset) => (
            <Button
              key={preset}
              size="sm"
              variant="ghost"
              onPress={() => setValue(String(preset))}
            >
              ${preset}
            </Button>
          ))}
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <span className="text-body">Resets every</span>
        <Segmented
          label="Resets every"
          size="md"
          options={LIMIT_PERIODS}
          value={period ?? ""}
          onChange={(next) => setPeriod(next as LimitPeriod)}
        />
        {initialPeriod === null ? (
          <p className="text-sm text-muted">
            {currentPeriodWords(target.own?.ceiling.budget_duration_sec)} Choose
            one to change that.
          </p>
        ) : null}
        <p className="text-sm text-muted">
          A request that would go past the limit is refused until it resets, and
          the refusal shows in Sessions.
        </p>
        {target.own ? (
          <p className="text-sm text-muted">
            What they've spent this period still counts against the new limit.
          </p>
        ) : null}
      </div>
    </FormDialog>
  )
}
