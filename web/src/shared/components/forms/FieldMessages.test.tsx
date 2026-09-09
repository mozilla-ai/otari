import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { ControlField, FieldMessages } from "./FieldMessages"

/**
 * The line under a field, and the two things about it that a reader cannot see.
 *
 * The first is which element wears the type role. The docstring spends a
 * paragraph on it because getting it wrong compiles and ships: HeroUI merges a
 * className through tailwind-merge, which cannot tell a custom `text-caption`
 * from a text color, so a role and a color on the same element lose the role and
 * keep the color. The role therefore belongs on the wrapper and the child gets
 * nothing but its ink, and neither half of that is visible in a rendered tree
 * until somebody measures the text.
 *
 * The second is the reserve. It exists to stop a form jumping when a message
 * appears, so its failure mode is motion on somebody else's screen rather than
 * anything a test would notice, and it is expressed as the role's own
 * line-height variable rather than a pixel so a retune of the caption carries
 * the reserve with it.
 *
 * What jsdom can and cannot say here: it does no layout, so the reserve is
 * asserted as the class that carries it rather than as a measured height, the
 * same limit every size assertion in this suite works under.
 */
describe("FieldMessages", () => {
  const wrapper = (container: HTMLElement) =>
    container.firstElementChild as HTMLElement

  it("wears the caption role itself rather than putting it on the message", () => {
    // The load-bearing half: the role on the wrapper, the child left with only
    // its color. Reversed, tailwind-merge drops the role and the line renders at
    // HeroUI's own 12px, which is a point under the role and looks almost right.
    const { container } = render(
      <FieldMessages>
        <p className="text-muted">Something to say</p>
      </FieldMessages>,
    )
    expect(wrapper(container)).toHaveClass("text-caption")
    const message = screen.getByText("Something to say")
    expect(message).toHaveClass("text-muted")
    expect(message.className).not.toContain("text-caption")
  })

  it("holds a line open by default, so an arriving message moves nothing", () => {
    const { container } = render(
      <FieldMessages>
        <p className="text-muted">Something to say</p>
      </FieldMessages>,
    )
    // The variable, not a pixel: `min-h-[19px]` would be the same height today
    // and would stop tracking the role the first time the caption is retuned.
    expect(wrapper(container)).toHaveClass(
      "min-h-[var(--text-caption-step--line-height)]",
    )
    expect(wrapper(container).className).not.toContain("min-h-[19px]")
  })

  it("holds nothing open where the caller says the line never speaks", () => {
    // `reserve={false}` is for a field in a table row or a toolbar. Reserving
    // there would put a band of empty space through every row of a table.
    const { container } = render(
      <FieldMessages reserve={false}>
        <p className="text-muted">Something to say</p>
      </FieldMessages>,
    )
    expect(wrapper(container)).toHaveClass("text-caption")
    expect(wrapper(container).className).not.toContain("min-h-")
  })

  it("renders what it is given", () => {
    render(
      <FieldMessages>
        <p className="text-muted">Enter a number above zero</p>
      </FieldMessages>,
    )
    expect(screen.getByText("Enter a number above zero")).toBeInTheDocument()
  })
})

/**
 * The label and description rungs of a control that is not a HeroUI field.
 *
 * This exists because the construct was hand-written each time and was born at
 * the wrong size in eight files at once: a `text-sm` label over a `text-xs`
 * description, which puts the description a point under the caption role. So
 * the two rungs and the absent-description case are what is worth holding here.
 */
describe("ControlField", () => {
  it("puts the label on the body role, matching a HeroUI field's own", () => {
    // Measured against HeroUI's label rather than approximated: both render
    // 14px/20px at weight 400. A `text-sm` here would be the same size and
    // would leave the label outside the scale.
    render(<ControlField label="Model access" />)
    const label = screen.getByText("Model access")
    expect(label).toHaveClass("text-body")
    expect(label.className).not.toContain("text-sm")
  })

  it("reserves nothing when there is no description to reserve for", () => {
    // Not an opt-out from the reserve: a control with no description will never
    // say anything, so an empty line under it is space held for nothing.
    const { container } = render(<ControlField label="Model access" />)
    expect(container.querySelector(".text-caption")).toBeNull()
    expect(container.innerHTML).not.toContain("min-h-")
  })

  it("sends a description through the caption role and its reserve", () => {
    const { container } = render(
      <ControlField label="Model access" description="Narrow, never widen" />,
    )
    const line = container.querySelector(".text-caption") as HTMLElement
    expect(line).not.toBeNull()
    expect(line).toHaveClass("min-h-[var(--text-caption-step--line-height)]")
    // Same division as FieldMessages on its own: the role above, the ink below.
    const message = screen.getByText("Narrow, never widen")
    expect(message).toHaveClass("text-muted")
    expect(message.className).not.toContain("text-caption")
  })

  it("passes the caller's reserve down rather than deciding for itself", () => {
    const { container } = render(
      <ControlField
        label="Model access"
        description="Narrow, never widen"
        reserve={false}
      />,
    )
    const line = container.querySelector(".text-caption") as HTMLElement
    expect(line).not.toBeNull()
    expect(line.className).not.toContain("min-h-")
  })

  it("renders the control it labels", () => {
    render(
      <ControlField label="Model access">
        <button type="button">Pick models</button>
      </ControlField>,
    )
    expect(
      screen.getByRole("button", { name: "Pick models" }),
    ).toBeInTheDocument()
  })
})
