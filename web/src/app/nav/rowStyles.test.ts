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

  it("hovers a resting row one step off the rail, and focuses it with the ring alone", () => {
    // Hover takes a fill; focus deliberately does not. A row that can be
    // selected cannot also spend the fill channel on focus, and the ring is
    // what focus is drawn with everywhere. Both still brighten the ink, so a
    // focused row is not weaker than a hovered one, it is differently marked.
    expect(resting).toContain("hover:bg-surface-alt")
    expect(resting).toContain("hover:text-foreground")
    expect(resting).toContain("focus-visible:text-foreground")
    // The regression this replaces: a fill outlives the click that focused the
    // row where a hover does not outlive the pointer, so a group trigger
    // clicked open kept the fill and read as selected.
    expect(resting).not.toContain("focus-visible:bg-")
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

  it("marks a selected row with the quieter fill and an edge hover cannot borrow", () => {
    // The louder of the two rungs on purpose, which is the swap: selection used
    // to take `surface-alt` while hover took `surface-subtle`, the louder one,
    // so the pointer made any row look more current than the current one.
    expect(selected).toContain("bg-surface-subtle")
    expect(selected).toContain("text-foreground")
    // Two adjacent rungs are a fragile way to carry "which page am I on", so
    // the state also gets a structural channel no transient state paints.
    expect(selected).toContain("border-foreground")
    // The edge is reserved on every row and only colored when selected, which is
    // what stops selection moving anything. The compensating `pl-` this
    // replaces collided with a nested row's own indent, so selecting a child
    // dropped it back into its parent's lane.
    expect(resting).toContain("border-l-2")
    expect(resting).toContain("border-transparent")
    // Only one border-color utility may reach a row. Carrying both, with the
    // width in the base and the two colors in the states, compiles clean and
    // measures transparent on the selected row: they are the same utility at
    // the same specificity, so Tailwind's emitted order decides, and it emits
    // `border-transparent` last.
    expect(selected).not.toContain("border-transparent")
    expect(selected).not.toMatch(/(?<![\w:-])pl-/)
  })

  it("keeps a nested row's indent when it is the selected one", () => {
    const nested = navRowClass({ nested: true })
    const nestedSelected = navRowClass({ nested: true, isActive: true })
    for (const row of [nested, nestedSelected]) {
      expect(row).toContain("pl-[3.125rem]")
    }
    // No `dark:` override. The surface family sits one rung above the
    // background family in both themes, so `surface-alt` is a step off the
    // rail either way. An override here would resolve to the rail's own value
    // and erase the selection.
    expect(selected).not.toContain("dark:bg-")
  })

  it("gives the group holding the selected row ink and nothing else", () => {
    const ancestor = navRowClass({ ancestor: true })
    expect(ancestor).toContain("text-foreground")
    // Neither of the two channels that say "this is the row you are on". The
    // measured bug: an expanded group and its selected child were byte-identical
    // while only the child carried aria-current, so the page claimed two current
    // rows and the accessibility tree claimed one.
    expect(ancestor).not.toMatch(/(?<![\w:-])bg-/)
    expect(ancestor).toContain("border-transparent")
    expect(ancestor).not.toContain("border-foreground")
    // It still answers the pointer, unlike the selected row: clicking it
    // collapses the group, so there is something for an affordance to promise.
    expect(ancestor).toContain("hover:bg-surface-alt")
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
      // The ring's values are globals.css's, not this module's, so what is
      // asserted here is that a row asks for one at utility strength. Spelling
      // the width, offset and color back would recreate the disagreement that
      // moving them into one place resolved.
      expect(row).toContain("focus-visible:otari-focus-ring")
      // Folds HeroUI's own inner ring away, which only matters on the rows that
      // are HeroUI Buttons.
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
