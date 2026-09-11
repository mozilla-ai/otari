import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { Tooltip } from "@/design-system/overlays/Tooltip"

/**
 * The two trigger forms, and the reason there are two.
 *
 * HeroUI's trigger is a `div` it gives `role="button"` and `tabIndex=0`, which
 * is right for content that is not a control and wrong around one: a real
 * button inside it is announced twice, takes two tab stops, and gives every
 * `getByRole("button")` two matches. The function form spreads the trigger's
 * props onto the control instead, so there is one element playing both parts.
 */
describe("Tooltip", () => {
  it("wraps a plain node in the library's own trigger", () => {
    render(
      <Tooltip content="An exact timestamp">
        <span>2 hours ago</span>
      </Tooltip>,
    )
    const trigger = screen.getByRole("button")
    expect(trigger.tagName).toBe("DIV")
    expect(trigger).toHaveTextContent("2 hours ago")
  })

  it("hands a control the trigger's props instead of wrapping it", () => {
    const { container } = render(
      <Tooltip content="Delete">
        {(props) => (
          <button {...props} type="button" aria-label="Delete">
            x
          </button>
        )}
      </Tooltip>,
    )
    // One control, not a control inside a control. This is the assertion the
    // wrapper form fails: it renders a div reported as a button around a
    // button, so the name resolves twice.
    expect(screen.getAllByRole("button")).toHaveLength(1)
    expect(container.firstElementChild?.tagName).toBe("BUTTON")
    // And the props reached it, which is what opens the tooltip at all: a
    // render function that ignores them compiles, renders, and never opens.
    expect(container.firstElementChild).toHaveAttribute(
      "data-slot",
      "tooltip-trigger",
    )
  })
})
