import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { HeadroomRing } from "@/design-system/metrics/HeadroomRing"

/** The figure is the reading, so each promise the component makes is a figure. */
describe("HeadroomRing", () => {
  it.each([
    [0, "100% left"],
    // Any use at all shows, however large the limit.
    [1e-7, "99% left"],
    // Float noise: 1 - 0.9 is 0.0999…, which must not read "9% left".
    [0.9, "10% left"],
    // A cent from the cap has nothing left to speak of.
    [0.99998, "0% left"],
    [1, "0% left"],
    [1.37, "Over limit"],
    [Infinity, "Over limit"],
  ])("reads %s used as %s", (used, text) => {
    render(<HeadroomRing used={used} />)
    expect(screen.getByText(text)).toBeInTheDocument()
  })
})
