import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { TrendChip, trendState } from "./TrendChip"

// The arrow is aria-hidden decoration and which hue HeroUI paints is what the
// stories and the screenshot matrix are for, so the rendered assertions here
// stay on text, sign, and whether an arrow is there at all. The mapping from a
// fraction to a direction and a status color is tested as a function instead.
function hasArrow(container: HTMLElement): boolean {
  return container.querySelector("svg") !== null
}

describe("trendState", () => {
  it("reads the sign as the direction", () => {
    expect(trendState(0.033, "neutral").direction).toBe("up")
    expect(trendState(-0.021, "neutral").direction).toBe("down")
  })

  it("calls a change too small to print flat rather than a fall", () => {
    // -0.0004 rounds to "0.0%", and a down arrow beside that reads as a bug.
    expect(trendState(-0.0004, "neutral").direction).toBe("flat")
    expect(trendState(0, "neutral").direction).toBe("flat")
    expect(trendState(-0, "neutral").direction).toBe("flat")
  })

  it("treats NaN as flat rather than falling through to a fall", () => {
    expect(trendState(Number.NaN, "up-is-good")).toEqual({
      direction: "flat",
      color: "default",
    })
  })

  it("passes no judgment when the metric has no good direction", () => {
    expect(trendState(0.12, "neutral").color).toBe("default")
    expect(trendState(-0.12, "neutral").color).toBe("default")
  })

  it("colors by the metric's own axis, not the number's", () => {
    expect(trendState(0.033, "up-is-good").color).toBe("success")
    expect(trendState(-0.021, "up-is-good").color).toBe("danger")
    expect(trendState(-0.059, "down-is-good").color).toBe("success")
    expect(trendState(0.12, "down-is-good").color).toBe("danger")
  })

  it("leaves a flat change uncolored whatever the polarity", () => {
    expect(trendState(0, "up-is-good").color).toBe("default")
    expect(trendState(0, "down-is-good").color).toBe("default")
  })
})

describe("TrendChip", () => {
  it("renders nothing when there is no comparable previous value", () => {
    expect(
      render(<TrendChip fraction={null} />).container,
    ).toBeEmptyDOMElement()
    expect(
      render(<TrendChip fraction={undefined} />).container,
    ).toBeEmptyDOMElement()
  })

  it("signs a rise and shows an arrow", () => {
    const { container } = render(<TrendChip fraction={0.033} />)
    expect(screen.getByText("+3.3%")).toBeInTheDocument()
    expect(hasArrow(container)).toBe(true)
  })

  it("keeps the minus sign on a fall", () => {
    const { container } = render(<TrendChip fraction={-0.021} />)
    expect(screen.getByText("-2.1%")).toBeInTheDocument()
    expect(hasArrow(container)).toBe(true)
  })

  it("drops the arrow and prints a bare zero when nothing moved", () => {
    const { container } = render(<TrendChip fraction={-0.0004} />)
    expect(screen.getByText("0.0%")).toBeInTheDocument()
    expect(hasArrow(container)).toBe(false)
  })

  it("says the judgment, not just the direction, when there is one to make", () => {
    // The same fall under each polarity: identical text, identical arrow, and
    // the color is the only visible difference, so the announced phrase is all a
    // screen reader has to tell the two apart.
    const good = render(
      <TrendChip fraction={-0.021} polarity="down-is-good" />,
    ).container
    expect(good).toHaveTextContent("down, better")
    const bad = render(
      <TrendChip fraction={-0.021} polarity="up-is-good" />,
    ).container
    expect(bad).toHaveTextContent("down, worse")
  })

  it("says the direction alone when the metric has no good direction", () => {
    render(<TrendChip fraction={-0.021} polarity="neutral" />)
    expect(screen.getByText("down")).toBeInTheDocument()
  })

  it("names a flat change rather than leaving a bare number", () => {
    render(<TrendChip fraction={0} />)
    expect(screen.getByText("no change")).toBeInTheDocument()
  })

  it("takes an absolute value as its text and still reads the sign", () => {
    const { container } = render(<TrendChip fraction={0.184} text="+$1,234" />)
    expect(screen.getByText("+$1,234")).toBeInTheDocument()
    expect(hasArrow(container)).toBe(true)
  })

  it("reads the caption out with the number", () => {
    render(<TrendChip fraction={-0.059} caption="vs last month" />)
    expect(screen.getByText("-5.9% vs last month")).toBeInTheDocument()
  })

  it("sets figures tabular so a ticking chip keeps its width", () => {
    render(<TrendChip fraction={0.033} />)
    expect(screen.getByText("+3.3%").className).toContain("tabular-nums")
  })

  it("scales the arrow with the chip", () => {
    const { container } = render(<TrendChip fraction={0.033} size="lg" />)
    expect(container.querySelector("svg")?.getAttribute("class")).toContain(
      "size-4",
    )
  })

  it("passes a call site's layout classes through to the chip", () => {
    // Position, not restyling: a chip is placed by the row it sits in, and
    // without this the call site needs a wrapper element to do it.
    const { container } = render(
      <TrendChip fraction={0.033} className="ml-auto" />,
    )
    expect(container.firstElementChild).toHaveClass("ml-auto")
  })

  it("hides the arrow from the accessibility tree", () => {
    const { container } = render(<TrendChip fraction={0.033} />)
    expect(container.querySelector("svg")).toHaveAttribute(
      "aria-hidden",
      "true",
    )
  })
})
