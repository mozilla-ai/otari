import { Tooltip as HeroTooltip } from "@heroui/react"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
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
  afterEach(() => {
    vi.restoreAllMocks()
  })

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

  /**
   * The delay is read off the props HeroUI is handed rather than from an open
   * tooltip: react-aria's hover never fires under jsdom, so a test that waits
   * for the label waits forever whatever the delay is. Left unset, HeroUI falls
   * back to `--tooltip-delay` on the document root, which it ships at
   * react-aria's 1.5s warmup, and every icon action in every table goes back to
   * reading as unlabeled.
   */
  it("opens on its own delay rather than HeroUI's 1.5s default", () => {
    const root = vi.spyOn(HeroTooltip, "Root")
    render(
      <Tooltip content="Delete">
        <span>x</span>
      </Tooltip>,
    )
    // The exact value: `design/overlays.md` states it, so moving it has to move
    // the sentence that documents it too.
    expect(root.mock.calls[0]?.[0]?.delay).toBe(300)
  })
})
