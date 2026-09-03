import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { DismissChip } from "./surface"

/**
 * A value somebody chose, with the way to unchoose it, and two claims worth
 * holding to.
 *
 * The first is a target-size claim: the docstring says the dismiss is "a real
 * 24px target holding a 12px glyph", the point being that the control reads
 * small and is not. That is the third such claim in this component and the
 * first two shipped untested, which is how a quiet control ends up quietly
 * unpressable.
 *
 * The second is that the accessible name is BUILT rather than passed. Three
 * call sites all pass `dismissLabel` today, so the fallback that concatenates
 * a name out of the label and the value is the branch nothing exercises, and
 * it is the branch a fourth caller will land on.
 *
 * What jsdom can and cannot say here: it does no layout, so the sizes below are
 * asserted as the classes that carry them rather than as measured pixels. The
 * claim that 24 and 12 are what a browser paints is only provable in Playwright
 * (see web/e2e), and the same limit applies to every size assertion in this
 * suite.
 */
describe("DismissChip", () => {
  const dismiss = () => screen.getByRole("button")

  it("draws a target four times the area of the glyph inside it", () => {
    render(<DismissChip value="gpt-5.6" onDismiss={() => undefined} />)
    // 24px on the button, 12px on the mark. Two separate sizes on purpose: the
    // visible cross stays quiet while the thing a finger lands on is the size a
    // finger needs, so neither number may be changed to match the other.
    // Token checks, not substring: "h-6" is a substring of `max-h-64`, so a
    // substring assertion here passes for a button with no height at all.
    expect([...dismiss().classList]).toContain("h-6")
    expect([...dismiss().classList]).toContain("w-6")
    const glyph = dismiss().querySelector("svg") as SVGElement
    expect([...glyph.classList]).toContain("h-3")
    expect([...glyph.classList]).toContain("w-3")
  })

  it("hides the mark from assistive tech, which reads the button instead", () => {
    render(<DismissChip value="gpt-5.6" onDismiss={() => undefined} />)
    expect(dismiss().querySelector("svg")).toHaveAttribute(
      "aria-hidden",
      "true",
    )
  })

  it("dismisses", async () => {
    const user = userEvent.setup()
    const onDismiss = vi.fn()
    render(<DismissChip value="gpt-5.6" onDismiss={onDismiss} />)
    await user.click(dismiss())
    expect(onDismiss).toHaveBeenCalledTimes(1)
  })

  it("shows the value, and the dimension before it when there is one", () => {
    const { rerender } = render(
      <DismissChip value="gpt-5.6" onDismiss={() => undefined} />,
    )
    expect(screen.getByText("gpt-5.6")).toBeInTheDocument()
    // Nothing at all before the value, not an empty dimension: rendering the
    // divider unconditionally leaves a bare colon in front of every chip that
    // has no dimension, which is most of them.
    expect(screen.queryByText("Model:")).toBeNull()
    expect(screen.queryByText(":")).toBeNull()

    rerender(
      <DismissChip label="Model" value="gpt-5.6" onDismiss={() => undefined} />,
    )
    // The colon is part of the divider between dimension and value, not
    // punctuation the caller supplies, so a caller cannot spell it two ways.
    expect(screen.getByText("Model:")).toBeInTheDocument()
  })

  it("names the target from the value when the caller says nothing", () => {
    render(<DismissChip value="gpt-5.6" onDismiss={() => undefined} />)
    expect(
      screen.getByRole("button", { name: "Remove gpt-5.6" }),
    ).toBeInTheDocument()
  })

  it("names the dimension too when there is one", () => {
    // A multi-value filter renders one of these per value, so a name that said
    // only "Remove" would give a row of controls that are indistinguishable by
    // name, which is the whole navigation for a screen-reader user.
    render(
      <DismissChip label="Model" value="gpt-5.6" onDismiss={() => undefined} />,
    )
    expect(
      screen.getByRole("button", { name: "Remove Model gpt-5.6" }),
    ).toBeInTheDocument()
  })

  it("lets a caller override the built name entirely", () => {
    // All three call sites do this today. The built name is the fallback, and
    // an override has to win outright rather than be appended to.
    render(
      <DismissChip
        label="Model"
        value="gpt-5.6"
        dismissLabel="Remove Model filter"
        onDismiss={() => undefined}
      />,
    )
    expect(
      screen.getByRole("button", { name: "Remove Model filter" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /gpt-5\.6/ })).toBeNull()
  })
})
