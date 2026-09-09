import { type ReactNode, useEffect, useRef } from "react"

/**
 * Marks its subtree `data-scrolled` while the table inside it is scrolled off
 * its left edge, so the pinned lane can draw its boundary only then.
 *
 * A cue for a state disappears with the state: at rest a table has no internal
 * verticals, because its columns are not regions, which is exactly what
 * separates it from the KPI strip. The same principle the mid-column clip
 * follows.
 *
 * It reaches for `.table__scroll-container` because that element is HeroUI's
 * and no call site can put a listener on it any other way.
 */
export function TableScrollFrame({
  className,
  children,
}: {
  className: string
  children: ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const root = ref.current
    if (!root) return
    let scroller: HTMLElement | null = null
    const sync = () => {
      if (!scroller) return
      root.dataset.scrolled = scroller.scrollLeft > 0 ? "true" : "false"
    }
    const attach = () => {
      const found = root.querySelector<HTMLElement>(".table__scroll-container")
      if (!found || found === scroller) return
      scroller?.removeEventListener("scroll", sync)
      scroller = found
      sync()
      scroller.addEventListener("scroll", sync, { passive: true })
    }
    attach()
    // The scroller is HeroUI's and is not in the tree when this first runs, and
    // it is replaced when the table's body remounts. Watching for it costs one
    // observer per table; the alternative is an effect with no dependency
    // array, which re-queries and swaps the listener on every render of every
    // table in the app, sort and hover included.
    const observer = new MutationObserver(attach)
    observer.observe(root, { childList: true, subtree: true })
    return () => {
      observer.disconnect()
      scroller?.removeEventListener("scroll", sync)
    }
  }, [])
  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  )
}
