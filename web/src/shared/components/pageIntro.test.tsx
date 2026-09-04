import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { PageIntro } from "./surface"

/**
 * The page opening, pinned before eight hand-rolled copies converge on it.
 *
 * Every one of those copies spells this markup out, and they have already
 * drifted: three different bottom paddings, and each spelling 28/34 at semibold
 * as literal utilities where the scale now has a step. (Described rather than
 * quoted, for the reason `PageIntro`'s own docstring gives: the whole-tree rule
 * matches raw file contents and a test file is only excluded from it today.)
 * Converging them is only mechanical while the thing they converge onto is
 * fixed, so the four facts a copy could quietly change (the outline level, the
 * type token, the band's own padding, and the measure) are asserted here
 * rather than left to review.
 */
describe("PageIntro", () => {
  // The header element itself, taken from the tree rather than by role: in the
  // product it renders inside `main`, where `header` carries no banner role.
  const header = (container: HTMLElement) =>
    container.firstElementChild as HTMLElement

  it("opens the page's outline at level one", () => {
    // The one heading on the page that names the page. A hand-rolled copy that
    // reaches for a styled `div` loses the whole outline for a reader
    // navigating by heading, and looks identical.
    render(<PageIntro title="API keys" />)
    expect(
      screen.getByRole("heading", { level: 1, name: "API keys" }),
    ).toBeInTheDocument()
  })

  it("takes its type from the scale rather than from numbers", () => {
    // 28/34 at the semibold step lives in the token now. The eight copies each
    // wrote it as literals, and the literals asked for a weight the axis does
    // not have, so they rendered a step away from what they said.
    render(<PageIntro title="API keys" />)
    const heading = screen.getByRole("heading", { level: 1 })
    expect(heading.className).toBe("text-display")
  })

  it("carries the band's own bottom padding", () => {
    // `pb-5` on the header rather than a gap on the page, so no air is added
    // above the first rule. This is the value the hand-rolled copies drifted
    // away from, in three directions.
    expect(
      render(<PageIntro title="API keys" />).container.firstElementChild,
    ).toHaveClass("pb-5")
  })

  it("stacks on a narrow viewport and turns into a row above it", () => {
    const { container } = render(
      <PageIntro
        title="API keys"
        action={<button type="button">New key</button>}
      />,
    )
    expect([...header(container).classList]).toContain("flex-col")
    expect([...header(container).classList]).toContain("sm:flex-row")
    expect([...header(container).classList]).toContain("sm:items-start")
  })

  it("describes the page when given a description", () => {
    render(<PageIntro title="API keys">Keys authenticate requests.</PageIntro>)
    const description = screen.getByText("Keys authenticate requests.")
    expect(description.tagName).toBe("P")
    expect([...description.classList]).toContain("text-muted")
  })

  it("renders no paragraph at all when there is nothing to say", () => {
    // Not an empty one: a blank `p` still occupies its margin and pushes the
    // first rule down on exactly the pages with no description.
    const { container } = render(<PageIntro title="API keys" />)
    expect(container.querySelector("p")).toBeNull()
  })

  it("holds the title and description to a measure", () => {
    render(<PageIntro title="API keys">Keys authenticate requests.</PageIntro>)
    const column = screen.getByText("Keys authenticate requests.")
      .parentElement as HTMLElement
    expect([...column.classList]).toContain("max-w-[620px]")
  })

  it("narrows the description without dropping what the role already gives it", () => {
    // `descriptionClassName` is how the one page that needs a narrower measure
    // gets it, and it is the reason that page does not need a hand-rolled
    // header. It has to append: replacing the base would take the muted color
    // and the type with it, and the caller would then re-spell both.
    render(
      <PageIntro title="Guide" descriptionClassName="max-w-[560px]">
        How to use the gateway.
      </PageIntro>,
    )
    const description = screen.getByText("How to use the gateway.")
    expect([...description.classList]).toContain("max-w-[560px]")
    expect([...description.classList]).toContain("text-sm")
    expect([...description.classList]).toContain("text-muted")
  })

  it("puts the action beside the opening and refuses to let it shrink", () => {
    const { container } = render(
      <PageIntro
        title="API keys"
        action={<button type="button">New key</button>}
      >
        Keys authenticate requests.
      </PageIntro>,
    )
    const action = screen.getByRole("button", { name: "New key" })
      .parentElement as HTMLElement
    expect([...action.classList]).toContain("shrink-0")
    expect(action.parentElement).toBe(header(container))
  })

  it("leaves no empty action slot when a page has no page-level action", () => {
    // Half the pages have none, and an empty flex child there is a silent
    // second column that the `justify-between` pushes the title away from.
    const { container } = render(<PageIntro title="API keys" />)
    expect(header(container).children).toHaveLength(1)
  })
})
