import { useEffect, useRef, useState } from "react"

/**
 * A rule's name. On a desk it is one line, and one cut short by its lane
 * carries its whole name as the pointer's tooltip; the row stays one press
 * target, which a focusable tooltip trigger inside it would take over. On a
 * phone, where nothing hovers, the card's title wraps instead.
 */
export function RuleName({ name }: { name: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const [isCut, setCut] = useState(false)
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new name is new text in the same lane, which only a fresh measure sees.
  useEffect(() => {
    const node = ref.current
    if (!node || typeof ResizeObserver === "undefined") return
    const check = () => setCut(node.scrollWidth > node.clientWidth)
    const observer = new ResizeObserver(check)
    observer.observe(node)
    // A new name, or a web font that lands after the first layout, changes
    // the text without resizing the lane, which the observer does not see.
    check()
    void document.fonts?.ready.then(check)
    return () => observer.disconnect()
  }, [name])
  return (
    <span
      ref={ref}
      title={isCut ? name : undefined}
      className="block min-w-0 text-body md:truncate"
    >
      {name}
    </span>
  )
}
