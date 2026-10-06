import { type ReactNode, useState } from "react"
import { Field } from "@/design-system/forms/Field"

const MIDNIGHT = "00:00"

/**
 * A key's expiry as a date and a time, with the time already at midnight.
 *
 * `value` is the datetime-local shape ("YYYY-MM-DDTHH:mm", local time), or ""
 * for no expiry. It is two inputs rather than one `datetime-local` because
 * that input reports "" until every segment is filled, so picking only a date
 * would silently create a key that never expires. The time is kept while no
 * date is set, so it can be chosen in either order.
 */
export function ExpiryFields({
  label,
  value,
  onChange,
  description,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  description: ReactNode
}) {
  const [date, time] = value ? value.split("T") : ["", undefined]
  const [pendingTime, setPendingTime] = useState(MIDNIGHT)
  const shownTime = time ?? pendingTime

  const changeDate = (next: string) => {
    if (!next) setPendingTime(shownTime)
    onChange(next ? `${next}T${shownTime || MIDNIGHT}` : "")
  }

  const changeTime = (next: string) => {
    setPendingTime(next)
    if (date) onChange(`${date}T${next || MIDNIGHT}`)
  }

  return (
    <div className="flex max-w-md gap-3">
      <div className="min-w-0 flex-1">
        <Field
          label={label}
          value={date}
          onChange={changeDate}
          type="date"
          description={description}
          shouldReserveMessage
        />
      </div>
      <div className="min-w-0 flex-1">
        <Field
          label="Time"
          value={shownTime}
          onChange={changeTime}
          type="time"
          description="Midnight unless changed."
          shouldReserveMessage
        />
      </div>
    </div>
  )
}
