import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TraceTreeView } from "@/features/traces/TraceTreeView"
import type { Observation, Trace } from "@/features/traces/types"

function obs(
  id: string,
  parentObservationId: string | null = null,
): Observation {
  return {
    id,
    parentObservationId,
    type: "SPAN",
    name: id,
    startMs: 0,
    endMs: 10,
    level: "DEFAULT",
  }
}

function trace(observations: Observation[], id = "session-1"): Trace {
  return { id, name: id, timestamp: "2026-10-27T10:00:00Z", observations }
}

function Tree({
  value,
  collapsed = new Set(),
  query = "",
}: {
  value: Trace
  collapsed?: ReadonlySet<string>
  query?: string
}) {
  return (
    <TraceTreeView
      trace={value}
      selectedId={null}
      onSelect={() => {}}
      collapsed={collapsed}
      onToggle={() => {}}
      query={query}
    />
  )
}

const rowOf = (name: string) =>
  screen.getByRole("button", { name: new RegExp(`^Span, ${name},`) })
    .parentElement as HTMLElement

const arriving = (name: string) =>
  rowOf(name).classList.contains("otari-trace-arrive")

describe("TraceTreeView arrivals", () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it("does not animate the rows already there when the panel opens", () => {
    render(<Tree value={trace([obs("turn-0"), obs("request", "turn-0")])} />)

    expect(arriving("turn-0")).toBe(false)
    expect(arriving("request")).toBe(false)
  })

  it("animates the rows a later poll adds, in tree order", () => {
    const first = [obs("turn-0"), obs("request", "turn-0")]
    const { rerender } = render(<Tree value={trace(first)} />)

    rerender(
      <Tree
        value={trace([
          ...first,
          obs("turn-1"),
          obs("model", "turn-1"),
          obs("tool", "turn-1"),
        ])}
      />,
    )

    expect(arriving("turn-0")).toBe(false)
    expect(arriving("request")).toBe(false)
    expect(
      ["turn-1", "model", "tool"].map((name) =>
        rowOf(name).style.getPropertyValue("--arrive-delay"),
      ),
    ).toEqual(["0ms", "45ms", "90ms"])
  })

  it("caps the stagger so a large batch still lands quickly", () => {
    const { rerender } = render(<Tree value={trace([obs("turn-0")])} />)
    const batch = Array.from({ length: 10 }, (_, i) => obs(`step-${i}`))

    rerender(<Tree value={trace([obs("turn-0"), ...batch])} />)

    expect(rowOf("step-9").style.getPropertyValue("--arrive-delay")).toBe(
      "270ms",
    )
  })

  it("does not animate a session's first spans arriving after its summary", () => {
    const { rerender } = render(<Tree value={trace([])} />)

    rerender(<Tree value={trace([obs("turn-0"), obs("request", "turn-0")])} />)

    expect(arriving("turn-0")).toBe(false)
    expect(arriving("request")).toBe(false)
  })

  it("does not animate another session the panel switches to", () => {
    const { rerender } = render(<Tree value={trace([obs("turn-0")])} />)

    rerender(<Tree value={trace([obs("other")], "session-2")} />)

    expect(arriving("other")).toBe(false)
  })

  it("does not announce the first turn row, which wraps the steps already there", () => {
    const { rerender } = render(
      <Tree value={trace([obs("step-a"), obs("step-b")])} />,
    )

    rerender(
      <Tree
        value={trace([
          obs("turn-0"),
          obs("step-a", "turn-0"),
          obs("step-b", "turn-0"),
          obs("turn-1"),
          obs("step-c", "turn-1"),
        ])}
      />,
    )

    expect(arriving("turn-0")).toBe(false)
    expect(arriving("step-a")).toBe(false)
    expect(rowOf("turn-1").style.getPropertyValue("--arrive-delay")).toBe("0ms")
    expect(rowOf("step-c").style.getPropertyValue("--arrive-delay")).toBe(
      "45ms",
    )
  })

  it("lets a batch still entering finish when the next one lands", () => {
    vi.useFakeTimers()
    const { rerender } = render(<Tree value={trace([obs("turn-0")])} />)
    rerender(<Tree value={trace([obs("turn-0"), obs("turn-1")])} />)
    act(() => {
      vi.advanceTimersByTime(500)
    })

    rerender(
      <Tree value={trace([obs("turn-0"), obs("turn-1"), obs("turn-2")])} />,
    )

    expect(arriving("turn-1")).toBe(true)
    expect(rowOf("turn-1").style.getPropertyValue("--arrive-delay")).toBe("0ms")
    expect(rowOf("turn-2").style.getPropertyValue("--arrive-delay")).toBe("0ms")
  })

  it("does not animate a row that arrived under a collapsed parent when it is expanded", () => {
    const first = [obs("turn-0")]
    const later = [...first, obs("turn-1"), obs("request", "turn-1")]
    const hidden = new Set(["turn-1"])
    const { rerender } = render(
      <Tree value={trace(first)} collapsed={hidden} />,
    )
    rerender(<Tree value={trace(later)} collapsed={hidden} />)

    rerender(<Tree value={trace(later)} collapsed={new Set()} />)

    expect(arriving("turn-1")).toBe(true)
    expect(arriving("request")).toBe(false)
  })

  it("does not animate a row that arrived while a search hid the tree", () => {
    const first = [obs("turn-0")]
    const later = [...first, obs("turn-1")]
    const { rerender } = render(<Tree value={trace(first)} />)
    rerender(<Tree value={trace(later)} query="turn" />)

    rerender(<Tree value={trace(later)} />)

    expect(arriving("turn-1")).toBe(false)
  })

  it("stops marking rows as arrivals once their entrance has played", () => {
    vi.useFakeTimers()
    const { rerender } = render(<Tree value={trace([obs("turn-0")])} />)
    rerender(<Tree value={trace([obs("turn-0"), obs("turn-1")])} />)
    expect(arriving("turn-1")).toBe(true)

    act(() => {
      vi.advanceTimersByTime(1_000)
    })

    expect(arriving("turn-1")).toBe(false)
  })
})
