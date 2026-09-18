import { useEffect, useState } from "react"

import { formatElapsed } from "./activityModel"

// The wait, ticking between the 2s polls: on an entry whose whole point is that it
// has not finished, a number that only moved when a response landed would read as
// stalled. Rendered only inside the open live list, so nothing ticks on a page
// whose operator has not asked to watch one.
export function InFlightWait({ startedAtMs }: { startedAtMs: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])
  // Never negative: a poll can resolve between the tick and the paint.
  return (
    <span className="tabular-nums">
      {formatElapsed(Math.max(0, now - startedAtMs))}
    </span>
  )
}
