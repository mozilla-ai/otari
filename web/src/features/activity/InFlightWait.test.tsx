import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { InFlightWait } from "./InFlightWait"

afterEach(() => {
  vi.useRealTimers()
})

describe("InFlightWait", () => {
  it("ticks between the polls, so a running request does not read as stalled", () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const startedAtMs = Date.now() - 3_000
    render(<InFlightWait startedAtMs={startedAtMs} />)
    expect(screen.getByText("3s")).toBeInTheDocument()

    act(() => {
      vi.advanceTimersByTime(4_000)
    })
    expect(screen.getByText("7s")).toBeInTheDocument()
  })

  it("never shows a negative wait", () => {
    // A poll can resolve between the tick and the paint, putting the start
    // instant in the future.
    render(<InFlightWait startedAtMs={Date.now() + 5_000} />)
    expect(screen.getByText("0s")).toBeInTheDocument()
  })

  it("stops ticking once it is unmounted", () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const { unmount } = render(<InFlightWait startedAtMs={Date.now()} />)
    unmount()

    expect(vi.getTimerCount()).toBe(0)
  })
})
