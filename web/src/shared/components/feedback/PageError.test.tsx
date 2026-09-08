import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { PageError } from "@/shared/components/feedback/PageError"

describe("PageError", () => {
  it("renders the caught value through the banner", () => {
    render(<PageError error={new Error("the gateway said no")} />)

    expect(screen.getByRole("alert")).toHaveTextContent("the gateway said no")
  })

  it("omits the paragraph when there is nothing to add", () => {
    const { container } = render(<PageError error={new Error("bare")} />)

    expect(container.querySelector("p")).toBeNull()
  })

  it("renders what to do next beside the banner", () => {
    render(
      <PageError error={new Error("bare")}>Reload to try again.</PageError>,
    )

    expect(screen.getByText("Reload to try again.")).toBeInTheDocument()
  })
})
