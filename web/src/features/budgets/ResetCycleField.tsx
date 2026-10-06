import { CheckboxGroup } from "@/design-system/forms/CheckboxGroup"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"

import {
  CYCLE_OPTIONS,
  type CycleDraft,
  cycleDraftError,
  isIntervalCycle,
  MONTH_DAY_OPTIONS,
  MONTH_OPTIONS,
  WEEKDAYS,
} from "./resetCycle"

// How often a budget resets, and the settings the chosen cycle needs.
//
// One control rather than six fields on the form, because a cycle and its
// settings are one decision: picking Weekly and leaving the weekdays blank is
// not a half-filled form, it is a budget the server refuses. The settings swap
// with the cycle and nothing from the previous one is left on screen, which is
// the visible half of the same rule the write path follows.
//
// The pieces are all design-system primitives. Nothing here draws a control:
// `design/forms.md`'s tree answers each one, and a date is the native input
// `Field` already wraps.

const WEEKDAY_OPTIONS = WEEKDAYS.map((day) => ({
  value: String(day.bit),
  label: day.label,
}))

export function ResetCycleField({
  value,
  onChange,
  isDisabled,
}: {
  value: CycleDraft
  onChange: (next: CycleDraft) => void
  isDisabled?: boolean
}) {
  const set = (changes: Partial<CycleDraft>) =>
    onChange({ ...value, ...changes })
  const error = cycleDraftError(value)
  const isInterval = isIntervalCycle(value.cycle)

  return (
    <div className="flex flex-col gap-3">
      <Select
        label="Reset cycle"
        value={value.cycle}
        onChange={(next) => set({ cycle: next })}
        options={CYCLE_OPTIONS}
        isDisabled={isDisabled}
        description="Every cycle resets at 00:00 UTC."
      />

      {isInterval ? (
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="min-w-0 flex-1">
            <Field
              label={value.cycle === "every_n_hours" ? "Hours" : "Days"}
              value={value.everyN}
              onChange={(next) => set({ everyN: next })}
              placeholder={value.cycle === "every_n_hours" ? "6" : "14"}
              isDisabled={isDisabled}
              isInvalid={error !== undefined && value.everyN.trim() === ""}
              errorMessage={error}
              shouldReserveMessage
            />
          </div>
          <div className="min-w-0 flex-1">
            <Field
              label="Starting from"
              value={value.anchorAt}
              onChange={(next) => set({ anchorAt: next })}
              type="date"
              isDisabled={isDisabled}
              description="Later periods are counted from this date, so the cycle keeps its phase."
              shouldReserveMessage
            />
          </div>
        </div>
      ) : null}

      {value.cycle === "weekly" ? (
        <CheckboxGroup
          label="Reset on"
          value={value.weekdayBits}
          onChange={(next) => set({ weekdayBits: next })}
          options={WEEKDAY_OPTIONS}
          orientation="horizontal"
          isDisabled={isDisabled}
          isInvalid={error !== undefined}
          errorMessage={error}
          description="Pick several and the limit applies to each period between them, so Monday and Friday give a four-day period and a three-day one."
        />
      ) : null}

      {value.cycle === "monthly" ? (
        <Select
          label="Day of the month"
          value={value.monthDay}
          onChange={(next) => set({ monthDay: next })}
          options={MONTH_DAY_OPTIONS}
          isDisabled={isDisabled}
          description="Up to the 28th, so every month has the day."
        />
      ) : null}

      {value.cycle === "yearly" ? (
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="min-w-0 flex-1">
            <Select
              label="Month"
              value={value.month}
              onChange={(next) => set({ month: next })}
              options={MONTH_OPTIONS}
              isDisabled={isDisabled}
            />
          </div>
          <div className="min-w-0 flex-1">
            <Select
              label="Day"
              value={value.monthDay}
              onChange={(next) => set({ monthDay: next })}
              options={MONTH_DAY_OPTIONS}
              isDisabled={isDisabled}
            />
          </div>
        </div>
      ) : null}
    </div>
  )
}
