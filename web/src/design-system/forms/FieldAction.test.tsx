import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { FieldAction } from "./FieldAction"

/**
 * The reserve under a trailing action, and why an empty box is the point.
 *
 * A control row is laid out `items-end`, so each child's whole box is
 * bottom-aligned. A field's box carries the caption line it reserves, and a
 * bare button's does not, which puts the button a caption line below the input
 * it acts on. This holds the same line under the action so the two input lines
 * agree.
 *
 * jsdom does no layout, so the reserve is asserted as the class that carries
 * it, the same limit `FieldMessages.test.tsx` works under. What a test can say
 * here is that the reserve is present, empty, and reads its height from the
 * caption role rather than from a pixel.
 */
describe("FieldAction", () => {
  // Structural rather than by class: a token gets renamed, and a test that
  // breaks on a restyle teaches everyone to stop trusting the suite.
  const reserve = (container: HTMLElement) =>
    (container.firstElementChild as HTMLElement)
      .lastElementChild as HTMLElement | null

  it("holds a caption line under its child, so the action meets the input line", () => {
    const { container } = render(
      <FieldAction>
        <button type="button">Remove</button>
      </FieldAction>,
    )

    const line = reserve(container)
    expect(line).not.toBeNull()
    expect(line).toHaveTextContent("")
    // The variable rather than a pixel, so a retune of the caption carries the
    // action with it.
    expect(line).toHaveClass("min-h-[var(--text-caption-step--line-height)]")
  })

  it("renders its child and adds nothing above it", () => {
    const { container } = render(
      <FieldAction>
        <button type="button">Remove</button>
      </FieldAction>,
    )

    expect(screen.getByRole("button", { name: "Remove" })).toBeInTheDocument()
    // The action is first: a label rung here would push it off the input line
    // in the other direction.
    const wrapper = container.firstElementChild as HTMLElement
    expect(wrapper.firstElementChild).toBe(screen.getByRole("button"))
    expect(wrapper.children).toHaveLength(2)
  })
})
