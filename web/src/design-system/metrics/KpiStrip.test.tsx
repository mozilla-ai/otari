import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { KpiStrip } from "@/design-system/metrics/KpiStrip"

/** The grid's own element, which is the one inside `Section`'s band. */
function track(label: string): HTMLElement {
  return screen.getByText(label).parentElement as HTMLElement
}

describe("KpiStrip", () => {
  it("lays five tracks by default", () => {
    render(
      <KpiStrip empty={false}>
        <span>cell</span>
      </KpiStrip>,
    )

    expect(track("cell")).toHaveClass("xl:grid-cols-5")
  })

  it("narrows to the number of cells the caller passed", () => {
    // A strip whose track count outruns its children leaves a blank column at
    // the end of the row, which is what the tenant Overview would get where it
    // withholds the budget cell from a member.
    render(
      <KpiStrip empty={false} columns={4}>
        <span>cell</span>
      </KpiStrip>,
    )

    expect(track("cell")).toHaveClass("xl:grid-cols-4")
    expect(track("cell")).not.toHaveClass("xl:grid-cols-5")
  })
})
