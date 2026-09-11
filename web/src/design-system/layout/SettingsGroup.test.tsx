import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"

/**
 * The two rule tiers, pinned.
 *
 * A section rule divides the page and a row separator divides repeated things
 * inside one section, and the two are different weights on purpose. Using the
 * section tier for both flattens the hierarchy into a single weight, which is
 * what a hand-rolled list drifts into. Every settings list in the app is this
 * component and none of them names a tier, so the mistake is only reachable
 * from here.
 */
describe("SettingsGroup", () => {
  const rowsBand = () =>
    screen.getByText("a row").closest("section") as HTMLElement

  it("separates its rows on the subtle tier and bounds the group on the section tier", () => {
    render(
      <SettingsGroup title="Group">
        <div>a row</div>
      </SettingsGroup>,
    )
    const band = rowsBand()
    expect(band).toHaveClass("border-y")
    expect(band).toHaveClass("border-border")
    expect(band.className).not.toContain("border-border-subtle")

    const rows = band.firstElementChild as HTMLElement
    expect(rows).toHaveClass("divide-y")
    expect(rows).toHaveClass("divide-border-subtle")
  })

  it("introduces the group with a heading between rules", () => {
    render(
      <SettingsGroup title="Credential security" count={2}>
        <div>a row</div>
      </SettingsGroup>,
    )
    const heading = screen.getByRole("heading", { name: /Credential security/ })
    // 16px at the semibold step, which is what `text-title` is. Named as the
    // token rather than as numbers so this cannot drift from the scale.
    expect(heading).toHaveClass("text-title")
    expect(heading.textContent).toContain("(2)")
    // The rule above the heading is the group's opening; the rows band below
    // carries the one under it. Two bands is what puts the heading between them.
    expect(heading.closest("section")).toHaveClass("border-t")
  })

  it("drops the heading band entirely when there is nothing to put in it", () => {
    // A page filtered to one service already names it in the page title, and a
    // second heading there reads as two headings for one thing.
    render(
      <SettingsGroup>
        <div>a row</div>
      </SettingsGroup>,
    )
    expect(screen.queryByRole("heading")).toBeNull()
    expect(rowsBand()).toHaveClass("border-y")
  })

  it("caps its description to a readable measure", () => {
    // A band spans the page now. A sentence inside it should not: this is the
    // standing rule for prose in a full-bleed row, and it is carried here so a
    // caller does not have to remember it.
    render(
      <SettingsGroup title="Group" description="What this group is for.">
        <div>a row</div>
      </SettingsGroup>,
    )
    const description = screen.getByText("What this group is for.")
    expect(description).toHaveClass("max-w-prose")
  })

  it("puts an action on the heading row, in both shapes", () => {
    // The slot exists so a group that owns a collection carries the control
    // that adds to it, beside the words naming it. Both shapes, because the
    // heading band and the bounded header are two different returns.
    for (const bounded of [false, true]) {
      const { unmount } = render(
        <SettingsGroup
          bounded={bounded}
          title="Search tools"
          action={<button type="button">Add search tool</button>}
        >
          <div>a row</div>
        </SettingsGroup>,
      )
      const heading = screen.getByRole("heading", { name: "Search tools" })
      const action = screen.getByRole("button", { name: "Add search tool" })
      // The same row, which is what "on the heading row" means structurally:
      // one element holds both, rather than the action following the band.
      expect(heading.closest("div")?.parentElement).toContainElement(action)
      unmount()
    }
  })

  it("renders the heading row for an action alone, with no title", () => {
    // The branch this prop moved: the row used to be skipped whenever there
    // were no words, which would now also drop the control.
    render(
      <SettingsGroup action={<button type="button">Add</button>}>
        <div>a row</div>
      </SettingsGroup>,
    )
    expect(screen.queryByRole("heading")).toBeNull()
    expect(screen.getByRole("button", { name: "Add" })).toBeInTheDocument()
  })
})
