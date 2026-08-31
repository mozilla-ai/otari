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

  it("presses a resting row the other way off the rail", () => {
    // `active` covers the Links and the plain expand buttons; `data-pressed` is
    // what react-aria reports on the rows that are HeroUI Buttons.
    //
    // The fill is the page canvas, not a border token used as a fill. Hover
    // lifts away from the chrome and a press sinks onto the ground behind it,
    // so the two are told apart by direction rather than by degree.
    for (const variant of ["active", "data-[pressed]"]) {
      expect(resting).toContain(`${variant}:bg-background`)
      expect(resting).toContain(`${variant}:text-foreground`)
    }
  })

  it("lifts a selected row off the rail with one class in both themes", () => {
    expect(selected).toContain("bg-surface-alt")
    expect(selected).toContain("text-foreground")
    // No `dark:` override. The surface family sits one rung above the
    // background family in both themes, so `surface-alt` is a step off the
    // rail either way. An override here would resolve to the rail's own value
    // and erase the selection.
    expect(selected).not.toContain("dark:bg-")
  })

  it("does not answer the pointer on a selected row at all", () => {
    // Neither hover nor press. It is the current page, so clicking it is a
    // no-op and there is nothing for an affordance to promise. A hover fill
    // here would also collide with a hovered resting row, which already gets
    // `hover:text-foreground` as well as a fill, and losing "is the row under
    // my pointer the page I am on" costs more than losing the response.
    expect(selected).not.toContain("hover:bg-")
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
