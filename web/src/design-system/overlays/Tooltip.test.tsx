import { readFileSync } from "node:fs"
import { join } from "node:path"

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

  /**
   * A source sweep rather than a hover: react-aria's hover never fires under
   * jsdom, so a test that waits for the label to open waits forever whatever
   * the delay is. The prop is what is worth guarding. Without it HeroUI falls
   * back to `--tooltip-delay` on the document root, which it ships at
   * react-aria's 1.5s warmup, and every icon action in every table goes back to
   * reading as unlabeled. Same shape as the globals.css sweeps in
   * `styles/foundation.test.ts`: the file is the artifact under test.
   */
  it("sets its own open delay rather than inheriting HeroUI's 1.5s", () => {
    const source = readFileSync(
      join(process.cwd(), "src", "design-system", "overlays", "Tooltip.tsx"),
      "utf8",
    )
    expect(source).toMatch(/HeroTooltip\.Root[^>]*\bdelay=\{OPEN_DELAY_MS\}/)
    const declared = source.match(/const OPEN_DELAY_MS = (\d+)/)
    expect(declared).not.toBeNull()
    expect(Number(declared?.[1])).toBeLessThanOrEqual(500)
  })
})
