import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { PageError } from "@/design-system/feedback/PageError"

describe("PageError", () => {
  it("renders the caught value through the banner", () => {
    render(<PageError error={new Error("the gateway said no")} />)

    expect(screen.getByRole("alert")).toHaveTextContent("the gateway said no")
  })

  it("omits the paragraph when there is nothing to add", () => {
    const { container } = render(<PageError error={new Error("bare")} />)

    expect(container.querySelector("p")).toBeNull()
  })

  // The banner renders nothing for a falsy value, so every one of these would
  // otherwise be a panel with an explanation and no error in it. The list runs
  // past `null` and `undefined` deliberately: a nullish coalesce covers those
  // two, so a case picked from them passes whether or not the other four work.
  it.each([
    ["empty string", ""],
    ["zero", 0],
    ["false", false],
    ["NaN", Number.NaN],
    ["null", null],
    ["undefined", undefined],
  ])("still shows a banner when the thrown value was %s", (_name, thrown) => {
    render(<PageError error={thrown} />)

    expect(screen.getByRole("alert")).toHaveTextContent("Something went wrong")
  })

  it("renders what to do next beside the banner", () => {
    render(
      <PageError error={new Error("bare")}>Reload to try again.</PageError>,
    )

    expect(screen.getByText("Reload to try again.")).toBeInTheDocument()
  })
})
