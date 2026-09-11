import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { ScanBorder } from "./ScanBorder"

function band(): HTMLElement {
  // The band itself, which is the child's parent: this component renders one
  // element and the arc lives on its `::after`, which no test can read.
  const child = screen.getByText("waiting")
  const parent = child.parentElement
  if (parent === null) throw new Error("the band has no element")
  return parent
}

describe("ScanBorder", () => {
  it("runs the sweep while something is genuinely awaited", () => {
    render(
      <ScanBorder isActive>
        <span>waiting</span>
      </ScanBorder>,
    )

    expect(band()).toHaveClass("otari-scan-border")
  })

  it("leaves a plain hairline once there is nothing left to wait for", () => {
    // The band does not disappear, it stops: a finished wait is still a band.
    render(
      <ScanBorder isActive={false}>
        <span>waiting</span>
      </ScanBorder>,
    )

    expect(band()).not.toHaveClass("otari-scan-border")
    expect(band()).toHaveClass("border-border")
  })

  it("takes the failure ink from the tone, not from a second rule", () => {
    // The arc's color is a variable the caller sets, which is how a failure
    // turns the sweep red without the stylesheet knowing what a failure is.
    render(
      <ScanBorder isActive tone="danger">
        <span>waiting</span>
      </ScanBorder>,
    )

    expect(band()).toHaveClass("[--scan-ink:var(--color-danger)]")
    expect(band()).toHaveClass("border-danger")
  })
})
