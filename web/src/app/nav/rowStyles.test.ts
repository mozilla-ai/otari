import { describe, expect, it } from "vitest"
import { navRowClass } from "./rowStyles"

/**
 * The rail's six interaction states, asserted as the classes a reader of the
 * sidebar actually sees. These are strings, so nothing here needs a DOM; what it
 * buys is that the ramp in the module's own doc comment cannot drift from the
 * classes below it, which is the way a design's states usually come apart: one
 * state gets adjusted and the neighbor it was tuned against does not.
 */
describe("navRowClass", () => {
  const resting = navRowClass()
  const selected = navRowClass({ isActive: true })

  it("leaves a resting row unfilled and muted", () => {
    expect(resting).toContain("text-muted")
    expect(resting).not.toMatch(/(?<![\w:-])bg-/)
  })

  it("hovers and focuses a resting row onto the same step off the rail", () => {
    // One step, not two: the same fill for both means a focused row is never
    // weaker than a hovered one, which is what the design asks for.
    for (const variant of ["hover", "focus-visible"]) {
      expect(resting).toContain(`${variant}:bg-surface-subtle`)
      expect(resting).toContain(`${variant}:text-foreground`)
    }
  })

  it("presses a resting row one step further, for either kind of press", () => {
    // `active` covers the Links and the plain expand buttons; `data-pressed` is
    // what react-aria reports on the rows that are HeroUI Buttons.
    for (const variant of ["active", "data-[pressed]"]) {
      expect(resting).toContain(`${variant}:bg-border`)
      expect(resting).toContain(`${variant}:text-foreground`)
    }
  })

  it("lifts a selected row off the rail, in both themes", () => {
    expect(selected).toContain("bg-surface-alt")
    expect(selected).toContain("dark:bg-surface")
    expect(selected).toContain("text-foreground")
  })

  it("still answers the pointer on a selected row, and only with a hover", () => {
    expect(selected).toContain("hover:bg-surface")
    expect(selected).toContain("dark:hover:bg-background")
    // The current page has nowhere to press to, so the pressed step stops at
    // the rows that navigate somewhere.
    expect(selected).not.toContain("active:")
    expect(selected).not.toContain("data-[pressed]:")
  })

  it("keeps every row on the sans face, groups included", () => {
    // A group's row is a `Disclosure.Trigger` inside a `Disclosure.Heading`,
    // which is a real <h3>, and the base reset hands every heading the display
    // serif. Without the family named here, Routing and Tools come out in Zilla
    // Slab beside leaves in Mozilla Text.
    for (const row of [resting, selected]) {
      expect(row).toContain("font-sans")
    }
  })

  it("rings every row on keyboard focus, whatever else it is wearing", () => {
    for (const row of [resting, selected]) {
      expect(row).toContain("focus-visible:outline-2")
      expect(row).toContain("focus-visible:outline-offset-2")
      expect(row).toContain("focus-visible:outline-focus")
      // Names the style back and folds HeroUI's own ring away, both of which
      // only matter on the rows that are HeroUI Buttons.
      expect(row).toContain("focus-visible:outline-solid")
      expect(row).toContain("focus-visible:ring-0")
      expect(row).toContain("focus-visible:ring-offset-0")
    }
  })

  it("hands every row the pointer cursor, whatever element it is", () => {
    // Only the `Link` rows get the hand from the user agent. The group triggers
    // and the account control are `<button>`s, which Tailwind's reset leaves on
    // the default arrow, so half the rail answered the pointer differently from
    // the other half until this was named on the shared row.
    for (const row of [
      resting,
      selected,
      navRowClass({ nested: true }),
      navRowClass({ collapsed: true }),
    ]) {
      expect(row).toContain("cursor-pointer")
    }
  })

  it("keeps the 44px floor and the shared shape on every variant", () => {
    for (const row of [
      resting,
      selected,
      navRowClass({ nested: true }),
      navRowClass({ collapsed: true }),
    ]) {
      expect(row).toContain("min-h-11")
      expect(row).toContain("rounded-lg")
    }
  })
})
