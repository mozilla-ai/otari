import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { SettingsGroup } from "./surface"

/**
 * The two rule tiers, pinned.
 *
 * A section rule divides the page and a row separator divides repeated things
 * inside one section, and the two are different weights on purpose. Using the
 * section tier for both flattens the hierarchy into a single weight, which was
 * found three separate times by eye before this component existed: on the
 * Overview activity preview, on Settings, and on the tools page's own
 * hand-rolled list. Every settings list in the app is this component now, and
 * none of them names a tier, so the mistake is only reachable from here.
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
    expect(band.className).toContain("border-y")
    expect(band.className).toContain("border-border")
    expect(band.className).not.toContain("border-border-subtle")

    const rows = band.firstElementChild as HTMLElement
    expect(rows.className).toContain("divide-y")
    expect(rows.className).toContain("divide-border-subtle")
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
    expect(heading.className).toContain("text-title")
    expect(heading.textContent).toContain("(2)")
    // The rule above the heading is the group's opening; the rows band below
    // carries the one under it. Two bands is what puts the heading between them.
    expect(heading.closest("section")?.className).toContain("border-t")
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
    expect(rowsBand().className).toContain("border-y")
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
    expect(description.className).toContain("max-w-prose")
  })
})
