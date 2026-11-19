import { useEffect, useState } from "react"

/**
 * The time a page reads its windows at, held still between renders so a window
 * is the same query on every render, and moved on when it could have gone
 * stale: when the page asks (a new window picked) and when the tab comes back
 * into view. Without that, a Today opened at 09:00 and read again at 17:00
 * would set today until 17:00 against yesterday until 09:00, and a tab left
 * open past midnight would keep the old day.
 */
export function usePageClock(): [now: number, refresh: () => void] {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible") setNow(Date.now())
    }
    document.addEventListener("visibilitychange", onVisible)
    return () => document.removeEventListener("visibilitychange", onVisible)
  }, [])
  return [now, () => setNow(Date.now())]
}
