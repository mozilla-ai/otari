import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { StatCard } from "@/shared/components/deprecated/StatCard"

describe("StatCard", () => {
  it("renders its label and value", () => {
    render(<StatCard label="Tracked cost" value="$12.34" />)
    expect(screen.getByText("Tracked cost")).toBeInTheDocument()
    expect(screen.getByText("$12.34")).toBeInTheDocument()
  })

  it("fits its grid track and avoids double padding", () => {
    // min-w-0 lets the tile shrink to its grid track (a fixed min-width overflowed
    // and overlapped the neighbour at two-up on mobile); p-0 zeroes HeroUI's own
    // card padding so it does not stack with Card.Content's and double the height.
    const { container } = render(<StatCard label="Requests" value="0" />)
    // Assert on the rendered root element rather than HeroUI's internal ".card"
    // class, so a library-internal class rename can't silently break this.
    const root = container.firstElementChild!
    expect(root.className).toContain("min-w-0")
    expect(root.className).toContain("p-0")
  })

  it("puts the trend under the value, on one row with the hint", () => {
    render(
      <StatCard
        label="Tracked cost"
        value="$12.34"
        trend={<span>up 4.2%</span>}
        hint="3 unpriced"
      />,
    )
    const value = screen.getByText("$12.34")
    const trend = screen.getByText("up 4.2%")
    const hint = screen.getByText("3 unpriced")
    // Below the value, not sharing its row.
    expect(value.parentElement).not.toContainElement(trend)
    expect(
      value.compareDocumentPosition(trend) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    // On one row with the hint, and ahead of it, rather than stacked above it.
    expect(trend.parentElement).toBe(hint.parentElement)
    expect(
      trend.compareDocumentPosition(hint) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it("reserves the aside row's height for a charted tile that has nothing to say", () => {
    // Otherwise a tile whose only aside was its delta loses the row, and its
    // sparkline rides above its neighbors' in the same grid row. Asserted on
    // the class for the same reason as the tile's own min-w-0 above: jsdom does
    // no layout, so the reservation is only observable as the utility.
    const { container } = render(
      <StatCard
        label="Tokens"
        value="3.3M"
        chart={<svg aria-label="trend" />}
      />,
    )
    expect(container.querySelector(".min-h-10\\.5")).not.toBeNull()
  })

  it("reserves nothing for a tile with neither aside nor chart", () => {
    const { container } = render(
      <StatCard label="Avg latency" value="1.33 s" />,
    )
    expect(container.querySelector(".min-h-10\\.5")).toBeNull()
  })
})
