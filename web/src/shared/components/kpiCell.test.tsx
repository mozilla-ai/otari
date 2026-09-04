import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { KpiCell } from "./surface"

/**
 * The cell's own column, pinned because the overflow it prevents is invisible
 * until the data is long enough and jsdom reports no geometry to catch it.
 *
 * A grid gives itself an `auto` column, whose floor is the widest child's
 * min-content, and the subline never wraps. Measured on the running strip with
 * a 59-character subline, an `auto` column came out 356px inside a 283px cell
 * and everything sized to it (the sparkline included) painted into the cell
 * beside it, by 129px at 1728 and 218px at 1280.
 */
describe("KpiCell", () => {
  const cell = (container: HTMLElement) =>
    container.firstElementChild as HTMLElement

  it("bounds its own column so a long child cannot widen it", () => {
    const { container } = render(
      <KpiCell label="Cache hit rate" value="13.1%" subline="239.3k read" />,
    )
    expect(cell(container).className).toContain("grid-cols-[minmax(0,1fr)]")
    expect(cell(container).className).toContain("min-w-0")
  })

  it("keeps the subline on one line and truncates it", () => {
    const subline = "1,239,300.4k read · 70,000.0k written · 12,345 misses"
    const { container, getByTitle } = render(
      <KpiCell label="Cache hit rate" value="13.1%" subline={subline} />,
    )
    // One line, so no cell is taller than its neighbours, and the full string
    // stays reachable on the title rather than being lost to the ellipsis.
    const line = cell(container).children[2] as HTMLElement
    expect(line.className).toContain("text-nowrap")
    expect(getByTitle(subline).className).toContain("truncate")
  })
})
