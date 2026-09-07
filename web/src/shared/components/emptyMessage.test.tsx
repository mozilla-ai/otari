import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { EmptyMessage } from "./surface"

/**
 * The band that holds space for an absent chart.
 *
 * The rule these pin is a cascade rule rather than a layout one, and it cost a
 * regression to learn: `py-10` and a caller's `py-0` are both plain class
 * selectors, so the winner is whichever Tailwind emits last in the stylesheet
 * and not whichever the call site writes last. Measured in the built CSS,
 * `.py-10` comes after `.py-0`, so a caller cannot turn the padding off by
 * passing a class. The min-height therefore REPLACES the padding here.
 */
describe("EmptyMessage", () => {
  function band(container: HTMLElement): HTMLElement {
    return container.firstElementChild as HTMLElement
  }

  it("pads itself when it is not holding a height", () => {
    const { container } = render(<EmptyMessage>Nothing here.</EmptyMessage>)

    expect(band(container)).toHaveClass("py-10")
    expect(band(container).className).not.toContain("min-h-")
  })

  it("drops the padding when it holds a height, rather than fighting it", () => {
    const { container } = render(
      <EmptyMessage minHeightClass="min-h-[16rem]">No data.</EmptyMessage>,
    )

    expect(band(container)).toHaveClass("min-h-[16rem]")
    // The assertion that matters: not merely that py-0 is present, but that
    // py-10 is absent. Both classes together render as py-10.
    expect(band(container)).not.toHaveClass("py-10")
    expect(band(container).className).not.toContain("py-")
  })

  it("keeps the treatment the same either way", () => {
    // The size and ink are the point of the component being shared, so they
    // hold across both modes.
    const plain = render(<EmptyMessage>a</EmptyMessage>)
    const held = render(
      <EmptyMessage minHeightClass="min-h-[16rem]">a</EmptyMessage>,
    )

    for (const one of [plain, held]) {
      expect(band(one.container)).toHaveClass(
        "flex",
        "items-center",
        "justify-center",
        "px-4",
        "text-center",
        "text-sm",
        "text-muted",
      )
    }
  })

  it("renders what it is given", () => {
    const { getByText } = render(<EmptyMessage>No requests yet.</EmptyMessage>)

    expect(getByText("No requests yet.")).toBeInTheDocument()
  })
})
