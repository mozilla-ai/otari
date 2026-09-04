import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { Section } from "./surface"

/**
 * The band's two shapes, pinned because the wrong one is invisible until
 * something shares the row.
 *
 * `.otari-bleed` escapes to `100cqw`, which is `<main>`, not the element's
 * parent. A band nested in a column therefore paints across its neighbours,
 * and jsdom reports no geometry to catch that, so the class contract is what
 * is asserted here.
 */
describe("Section", () => {
  const band = (container: HTMLElement) =>
    container.firstElementChild as HTMLElement

  it("escapes to the scroll area and restores the column inside", () => {
    const { container } = render(
      <Section className="py-5" contentClassName="flex">
        rows
      </Section>,
    )
    expect(band(container).className).toContain("otari-bleed")
    expect(band(container).firstElementChild?.className).toContain("mx-auto")
  })

  it("stays inside its container when nested", () => {
    const { container } = render(
      <Section bleed={false} className="py-5" contentClassName="flex">
        rows
      </Section>,
    )
    expect(band(container).className).not.toContain("otari-bleed")
    expect(band(container).className).toContain("py-5")
    // No centered column either: the container it sits in is the column, so a
    // second inset would pull the content off the alignment of everything
    // else in that column.
    expect(band(container).firstElementChild?.className).not.toContain(
      "mx-auto",
    )
    expect(band(container).firstElementChild?.className).toContain("flex")
  })
})
