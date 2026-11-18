import { useState } from "react"
import { type SpendWindow, windowOf } from "@/features/spend/period"

/**
 * "This week": the last seven days, whole local days (`sessionsSearchFor`
 * opens Sessions on the same window). Read over one clock for the page's life,
 * so the queries keyed on it hold still.
 */
export function useWeek(): SpendWindow {
  const [now] = useState(() => Date.now())
  return windowOf("7d", now)
}
