import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ErrorBoundary } from "@/shared/components/feedback/ErrorBoundary"

function Throws(): never {
  throw new Error("the bootstrap said nothing about oauth")
}

function ThrowsNothing(): never {
  throw null
}

describe("ErrorBoundary", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders its children while nothing throws", () => {
    render(
      <ErrorBoundary>
        <p>the sign-in screen</p>
      </ErrorBoundary>,
    )

    expect(screen.getByText("the sign-in screen")).toBeInTheDocument()
  })

  it("shows the error instead of unmounting to a blank document", () => {
    // React logs a caught render error through console.error regardless of the
    // boundary, so the assertion is about what is on screen, not about silence.
    vi.spyOn(console, "error").mockImplementation(() => {})

    const { container } = render(
      <ErrorBoundary>
        <Throws />
      </ErrorBoundary>,
    )

    expect(screen.getByRole("alert")).toHaveTextContent(
      "the bootstrap said nothing about oauth",
    )
    expect(container).not.toBeEmptyDOMElement()
  })

  it("catches a falsy thrown value rather than re-rendering the thrower", () => {
    // A boundary keyed on the thrown value's truthiness renders the child
    // again, and React answers a second throw by blanking the document.
    vi.spyOn(console, "error").mockImplementation(() => {})

    const { container } = render(
      <ErrorBoundary>
        <ThrowsNothing />
      </ErrorBoundary>,
    )

    expect(screen.getByRole("alert")).toBeInTheDocument()
    expect(container).not.toBeEmptyDOMElement()
  })
})
