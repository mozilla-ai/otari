export type ExpiryParts = { date: string; time: string }

const EMPTY_EXPIRY: ExpiryParts = { date: "", time: "" }

export function emptyExpiryParts(): ExpiryParts {
  return { ...EMPTY_EXPIRY }
}

function toDatetimeLocal(iso: string | null): string {
  if (!iso) return ""
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ""
  const pad = (value: number) => String(value).padStart(2, "0")
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`
}

function expiryPartsFromLocal(value: string): ExpiryParts {
  if (!value) return emptyExpiryParts()
  const [date = "", time = ""] = value.split("T")
  return { date, time }
}

export function expiryPartsFromIso(value: string | null): ExpiryParts {
  return expiryPartsFromLocal(toDatetimeLocal(value))
}

function currentLocalExpiryParts(): ExpiryParts {
  return expiryPartsFromIso(new Date().toISOString())
}

function nextLocalExpiryMinute(): ExpiryParts {
  const nextMinute = new Date()
  nextMinute.setSeconds(0, 0)
  nextMinute.setMinutes(nextMinute.getMinutes() + 1)
  return expiryPartsFromIso(nextMinute.toISOString())
}

export function expiryValue(parts: ExpiryParts): string {
  return parts.date && parts.time ? `${parts.date}T${parts.time}` : ""
}

export function withExpiryDate(parts: ExpiryParts, date: string): ExpiryParts {
  if (!date) return { ...parts, date: "" }
  if (parts.time) return { date, time: parts.time }
  const current = currentLocalExpiryParts()
  return date === current.date
    ? nextLocalExpiryMinute()
    : { date, time: current.time }
}

export function withExpiryTime(parts: ExpiryParts, time: string): ExpiryParts {
  if (!time) return { ...parts, time: "" }
  return { date: parts.date || currentLocalExpiryParts().date, time }
}
