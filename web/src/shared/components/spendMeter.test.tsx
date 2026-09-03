import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { SpendMeter, spendState } from "./surface"

/**
 * The three states of a spend, and the fact that the bar and the figure beside
 * it read them from the same function. Before this the code had two states: a
 * bare `spent > allocated` ternary, so everything under the limit looked
 * identical and the first thing anyone learned was that they had gone past.
 */
describe("spendState", () => {
  it("calls a spend well inside its allocation on track", () => {
    expect(spendState(10, 100)).toBe("on-track")
    expect(spendState(79.99, 100)).toBe("on-track")
  })

  it("calls a spend past the threshold near the limit, up to the limit itself", () => {
    expect(spendState(80, 100)).toBe("near-limit")
    expect(spendState(99.99, 100)).toBe("near-limit")
    // Spending exactly the allocation has not exceeded it.
    expect(spendState(100, 100)).toBe("near-limit")
  })

  it("calls a spend past the allocation over", () => {
    expect(spendState(100.01, 100)).toBe("over")
  })

  it("is a share of the allocation, not an absolute", () => {
    // "Nearly out" means the same on a small budget and a large one, which is
    // why the threshold cannot be a number of dollars.
    expect(spendState(40, 50)).toBe("near-limit")
    expect(spendState(40, 5000)).toBe("on-track")
  })

  it("treats an allocation of zero as on track rather than dividing by it", () => {
    // A budget with no limit reaches here through a caller that has already
    // said so; what matters is that it cannot produce Infinity or NaN.
    expect(spendState(10, 0)).toBe("on-track")
  })
})

describe("SpendMeter", () => {
  const fills = () =>
    Array.from(screen.getByRole("progressbar").querySelectorAll("span")).map(
      (el) => el.className,
    )

  it("draws one accent fill while on track", () => {
    render(<SpendMeter spent={20} allocated={100} ariaLabel="Spend" />)
    const painted = fills().filter((c) => c.includes("bg-"))
    expect(painted).toHaveLength(1)
    expect(painted[0]).toContain("bg-accent")
  })

  it("draws two segments near the limit, accent to the threshold and danger past it", () => {
    render(<SpendMeter spent={90} allocated={100} ariaLabel="Spend" />)
    const painted = fills().filter((c) => c.includes("bg-"))
    // Two blocks, in this order, which is what makes the state legible without
    // hue: the overshoot is a separate block rather than a recolored bar.
    expect(painted).toHaveLength(2)
    expect(painted[0]).toContain("bg-accent")
    expect(painted[1]).toContain("bg-danger")
  })

  it("draws the whole bar danger once over", () => {
    render(<SpendMeter spent={140} allocated={100} ariaLabel="Spend" />)
    const painted = fills().filter((c) => c.includes("bg-"))
    expect(painted).toHaveLength(1)
    expect(painted[0]).toContain("bg-danger")
    expect(painted[0]).toContain("w-full")
  })

  it("keeps the widget well formed and still says how far past the limit it is", () => {
    // Two facts, two fields. `aria-valuenow` stays inside the declared range,
    // because a progressbar reporting 140 out of 100 is malformed; the real
    // share goes in `aria-valuetext`, which is the field for a value's human
    // reading. Carrying the overshoot in `valuenow` was the earlier version and
    // abused one field to say the other's thing.
    render(<SpendMeter spent={140} allocated={100} ariaLabel="Spend" />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "100")
    expect(bar).toHaveAttribute("aria-valuetext", "140% of limit — over budget")
  })

  it("names the share in words below the limit too", () => {
    render(<SpendMeter spent={31} allocated={100} ariaLabel="Spend" />)
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuetext",
      "31% of limit",
    )
  })
})
