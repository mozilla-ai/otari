/**
 * A budget's reset cycle, as the page reads and writes it.
 *
 * One module because the cycle is one concept carried by six fields, and a
 * surface that touches some of them and not the others writes a row the server
 * refuses: switching a weekly budget to monthly without clearing the weekday
 * mask is two valid-looking fields and an impossible budget.
 */

import type { BudgetResetCycle } from "@/client"

/**
 * The settings a cycle carries, as every budget shape on the wire holds them.
 *
 * Spelled out rather than `Pick`ed off the request type, because those fields
 * are optional there and every value here is always present: a write sends the
 * whole set, and an absent key would leave the previous cycle's setting in place
 * on a PATCH, which is the one thing the server refuses.
 */
export type CycleFields = {
  reset_cycle: BudgetResetCycle | null
  reset_every_n: number | null
  reset_anchor_at: string | null
  reset_weekdays: number | null
  reset_month_day: number | null
  reset_month: number | null
}

/** The same, as a response carries it: the cycle widens to a bare string. */
export type ReadCycleFields = Omit<CycleFields, "reset_cycle"> & {
  reset_cycle: string | null
}

/**
 * Every cycle field, so a caller can clear the set before writing the one it
 * means. Spelled once here rather than at each form, which is how a stale
 * weekday mask survives a cycle change.
 */
export const EMPTY_CYCLE: CycleFields = {
  reset_cycle: null,
  reset_every_n: null,
  reset_anchor_at: null,
  reset_weekdays: null,
  reset_month_day: null,
  reset_month: null,
}

/** What the cycle picker offers, in the order it offers them. */
export const CYCLE_OPTIONS: readonly {
  value: string
  label: string
}[] = [
  { value: "never", label: "Never" },
  { value: "every_n_hours", label: "Every N hours" },
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" },
  { value: "every_n_days", label: "Every N days" },
  { value: "monthly", label: "Monthly" },
  { value: "yearly", label: "Yearly" },
]

/**
 * Weekday labels in `date.weekday()` order, which is the order the mask's bits
 * are in: bit 0 is Monday.
 */
export const WEEKDAYS: readonly {
  bit: number
  label: string
  short: string
}[] = [
  { bit: 0, label: "Monday", short: "Mon" },
  { bit: 1, label: "Tuesday", short: "Tue" },
  { bit: 2, label: "Wednesday", short: "Wed" },
  { bit: 3, label: "Thursday", short: "Thu" },
  { bit: 4, label: "Friday", short: "Fri" },
  { bit: 5, label: "Saturday", short: "Sat" },
  { bit: 6, label: "Sunday", short: "Sun" },
]

/** The highest day a monthly or yearly cycle may land on; every month has it. */
export const MAX_MONTH_DAY = 28

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
]

/** The weekdays a mask names, as bit numbers, lowest first. */
export function weekdaysFromMask(mask: number | null): number[] {
  if (!mask) return []
  return WEEKDAYS.filter((day) => (mask & (1 << day.bit)) !== 0).map(
    (day) => day.bit,
  )
}

/** The mask naming these weekdays. */
export function maskFromWeekdays(bits: readonly number[]): number {
  return bits.reduce((mask, bit) => mask | (1 << bit), 0)
}

/** The ordinal an operator reads a day of the month as: 1st, 2nd, 3rd, 21st. */
export function ordinal(day: number): string {
  // 11th through 13th are the exception the last-digit rule gets wrong.
  if (day % 100 >= 11 && day % 100 <= 13) return `${day}th`
  const suffix = { 1: "st", 2: "nd", 3: "rd" }[day % 10] ?? "th"
  return `${day}${suffix}`
}

function weekdayPhrase(mask: number | null): string {
  const picked = weekdaysFromMask(mask)
  if (picked.length === 0) return "no day"
  const names = picked.map((bit) => WEEKDAYS[bit].label)
  if (names.length === 1) return names[0]
  // "Monday, Wednesday and Friday": the list separator an operator reads,
  // rather than the comma-only join a naive join produces.
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`
}

function plural(count: number, noun: string): string {
  return `${count} ${count === 1 ? noun : `${noun}s`}`
}

/**
 * How a reset cycle reads in a table cell or a detail view.
 *
 * Every cycle says UTC, because a budget that rolls at midnight somewhere else
 * is a different product and the operator reading this is deciding whether a
 * refusal was due.
 */
export function cycleLabel(budget: ReadCycleFields): string {
  switch (budget.reset_cycle) {
    case null:
    case undefined:
      return "Never"
    case "daily":
      return "Daily, at 00:00 UTC"
    case "weekly":
      return `Every ${weekdayPhrase(budget.reset_weekdays ?? null)} (UTC)`
    case "monthly":
      return `Monthly, the ${ordinal(budget.reset_month_day ?? 1)} at 00:00 UTC`
    case "yearly": {
      const month = MONTHS[(budget.reset_month ?? 1) - 1] ?? ""
      return `Yearly, ${month} ${ordinal(budget.reset_month_day ?? 1)} (UTC)`
    }
    case "every_n_hours":
      return `Every ${plural(budget.reset_every_n ?? 1, "hour")}`
    case "every_n_days":
      return `Every ${plural(budget.reset_every_n ?? 1, "day")}`
    default:
      // A cycle written by a newer gateway than this dashboard. Shown rather
      // than hidden, because a budget that resets on a cadence this page cannot
      // name still resets, and "Never" would be a lie.
      return budget.reset_cycle
  }
}

/**
 * The same cycle as a bare unit, for a label that reads as a rate.
 *
 * `cycleLabel` spells the cadence out because a table cell is where an operator
 * checks exactly when a budget turns over. A derived name is not: it has to fit
 * beside a figure (`$50.00 / month`), so it takes the shortest true phrase.
 * `undefined` where a budget never resets, which is a label with no rate at all
 * rather than one reading "never".
 */
export function shortCycleLabel(budget: ReadCycleFields): string | undefined {
  switch (budget.reset_cycle) {
    case null:
    case undefined:
      return undefined
    case "daily":
      return "day"
    case "weekly": {
      const picked = weekdaysFromMask(budget.reset_weekdays ?? null)
      // Several days are several periods of different lengths, so there is no
      // single unit to name: the count is the honest short form.
      return picked.length === 1 ? "week" : `${picked.length}x a week`
    }
    case "monthly":
      return "month"
    case "yearly":
      return "year"
    case "every_n_hours": {
      const n = budget.reset_every_n ?? 1
      return n === 1 ? "hour" : plural(n, "hour")
    }
    case "every_n_days": {
      const n = budget.reset_every_n ?? 1
      return n === 1 ? "day" : plural(n, "day")
    }
    default:
      return budget.reset_cycle
  }
}

/**
 * Which `CYCLE_OPTIONS` value a budget corresponds to, for the form.
 *
 * A budget that exists and resets on nothing is "never"; a budget that does not
 * exist yet opens on monthly, which is what the form defaulted to before the
 * cycles were richer. A new budget defaulting to "never" would be a cap nobody
 * asked for that never turns over.
 */
export function cycleValue(budget: ReadCycleFields | undefined): string {
  if (budget === undefined) return "monthly"
  return budget.reset_cycle ?? "never"
}

/** Whether a cycle takes an interval and an anchor. */
export function isIntervalCycle(cycle: string): boolean {
  return cycle === "every_n_hours" || cycle === "every_n_days"
}

/**
 * The fields a cycle carries, cleared of every other cycle's.
 *
 * The write path's whole job: the server refuses a budget holding a setting its
 * cycle does not take, so a form that edits one field has to send the set.
 */
export function cycleFields(
  cycle: string,
  settings: {
    everyN?: number | null
    anchorAt?: string | null
    weekdayBits?: readonly number[]
    monthDay?: number | null
    month?: number | null
  },
): CycleFields {
  if (cycle === "never") return { ...EMPTY_CYCLE }
  const base = { ...EMPTY_CYCLE, reset_cycle: cycle as BudgetResetCycle }
  if (isIntervalCycle(cycle)) {
    return {
      ...base,
      reset_every_n: settings.everyN ?? null,
      reset_anchor_at: settings.anchorAt ?? null,
    }
  }
  if (cycle === "weekly") {
    const mask = maskFromWeekdays(settings.weekdayBits ?? [])
    return { ...base, reset_weekdays: mask === 0 ? null : mask }
  }
  if (cycle === "monthly") {
    return { ...base, reset_month_day: settings.monthDay ?? null }
  }
  if (cycle === "yearly") {
    return {
      ...base,
      reset_month: settings.month ?? null,
      reset_month_day: settings.monthDay ?? null,
    }
  }
  return base
}

export const MONTH_OPTIONS = MONTHS.map((label, index) => ({
  value: String(index + 1),
  label,
}))

export const MONTH_DAY_OPTIONS = Array.from(
  { length: MAX_MONTH_DAY },
  (_unused, index) => ({ value: String(index + 1), label: ordinal(index + 1) }),
)

/**
 * The cycle as a form holds it: every field a string, because that is what an
 * input gives back and converting on each keystroke loses a half-typed number.
 */
export type CycleDraft = {
  cycle: string
  everyN: string
  anchorAt: string
  weekdayBits: string[]
  monthDay: string
  month: string
}

/** Today, as a date input spells it, for a cycle that needs an anchor and has none. */
function todayIso(): string {
  return new Date().toISOString().slice(0, 10)
}

/** The draft a stored budget opens the form on. */
export function cycleDraftFrom(
  budget: ReadCycleFields | undefined,
): CycleDraft {
  const anchor = budget?.reset_anchor_at
  return {
    cycle: cycleValue(budget),
    everyN: budget?.reset_every_n != null ? String(budget.reset_every_n) : "",
    anchorAt: anchor ? anchor.slice(0, 10) : todayIso(),
    weekdayBits: weekdaysFromMask(budget?.reset_weekdays ?? null).map(String),
    monthDay: String(budget?.reset_month_day ?? 1),
    month: String(budget?.reset_month ?? 1),
  }
}

/**
 * What the draft sends, cleared of every cycle's settings but its own.
 *
 * The anchor is sent as midnight UTC on the chosen day, because every cycle in
 * this product rolls at 00:00 UTC and an anchor carrying the browser's local
 * time would make an interval budget roll at an hour nobody picked.
 */
export function cycleFieldsFromDraft(draft: CycleDraft): CycleFields {
  const everyN = Number.parseInt(draft.everyN, 10)
  return cycleFields(draft.cycle, {
    everyN: Number.isFinite(everyN) ? everyN : null,
    anchorAt: draft.anchorAt ? `${draft.anchorAt}T00:00:00Z` : null,
    weekdayBits: draft.weekdayBits.map(Number),
    monthDay: Number.parseInt(draft.monthDay, 10) || null,
    month: Number.parseInt(draft.month, 10) || null,
  })
}

/**
 * Why this draft cannot be saved, or undefined when it can.
 *
 * The same rule the server enforces, said before the request rather than after:
 * each cycle needs its own settings, and the server answers 400 naming the field
 * where this names it on the control.
 */
export function cycleDraftError(draft: CycleDraft): string | undefined {
  if (isIntervalCycle(draft.cycle)) {
    const everyN = Number.parseInt(draft.everyN, 10)
    if (!Number.isFinite(everyN) || everyN < 1) {
      return "Enter how many, as a whole number of one or more."
    }
    if (!draft.anchorAt) return "Pick the date this cycle starts from."
  }
  if (draft.cycle === "weekly" && draft.weekdayBits.length === 0) {
    return "Pick at least one weekday."
  }
  return undefined
}
