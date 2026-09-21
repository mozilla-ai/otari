import { useEffect, useState } from "react"

/**
 * A value that settles after `delay` milliseconds of quiet.
 *
 * For a search term on its way into a query key. Typing straight into one
 * fetches per keystroke, and the answers race: the reply to `gp` can land after
 * the reply to `gpt`, leaving the list describing a term nobody is looking at.
 *
 * The first value is returned as it is, so a field that mounts with a term does
 * not sit empty for the delay. Every later change waits, and the timer is
 * cleared on the next change and on unmount, which is both halves of the leak
 * `performance.md` names: a timer outliving its component, and a second trigger
 * stacking one rather than replacing it.
 */
export function useDebounced<T>(value: T, delay = 250): T {
  const [settled, setSettled] = useState(value)

  useEffect(() => {
    if (value === settled) return
    const timer = setTimeout(() => setSettled(value), delay)
    return () => clearTimeout(timer)
  }, [value, settled, delay])

  return settled
}
