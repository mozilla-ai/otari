import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ErrorBoundary } from "@/design-system/feedback/ErrorBoundary"

function Thrower(): never {
  throw new Error("the bootstrap said nothing about oauth")
}

function throwing(value: unknown) {
  return function Throws(): never {
    throw value
  }
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
        <Thrower />
      </ErrorBoundary>,
    )

    expect(screen.getByRole("alert")).toHaveTextContent(
      "the bootstrap said nothing about oauth",
    )
    expect(container).not.toBeEmptyDOMElement()
  })

  it("clears the panel when the reset key changes", () => {
    // The panel would otherwise latch: App picks its branch from the hash, so a
    // throw on one public-auth link would survive following the next one.
    vi.spyOn(console, "error").mockImplementation(() => {})

    const { rerender } = render(
      <ErrorBoundary resetKey="#/verify-email">
        <Thrower />
      </ErrorBoundary>,
    )
    expect(screen.getByRole("alert")).toBeInTheDocument()

    rerender(
      <ErrorBoundary resetKey="#/reset-password">
        <p>the next link</p>
      </ErrorBoundary>,
    )

    expect(screen.queryByRole("alert")).toBeNull()
    expect(screen.getByText("the next link")).toBeInTheDocument()
  })

  it("keeps the panel while the reset key holds", () => {
    vi.spyOn(console, "error").mockImplementation(() => {})

    const { rerender } = render(
      <ErrorBoundary resetKey="#/verify-email">
        <Thrower />
      </ErrorBoundary>,
    )
    rerender(
      <ErrorBoundary resetKey="#/verify-email">
        <p>never reached</p>
      </ErrorBoundary>,
    )

    expect(screen.getByRole("alert")).toBeInTheDocument()
    expect(screen.queryByText("never reached")).toBeNull()
  })

  // Two hazards meet in a falsy throw, and every one of these values is legal to
  // throw. A boundary keyed on the thrown value's own truthiness renders the
  // child again, and React answers the second throw by blanking the document; a
  // banner keyed on it renders nothing, leaving a panel with no error in it.
  // Every case here exercises the first. Only the three below `undefined`
  // exercise the second, since a nullish coalesce already stood in for the two
  // above it, which is how that half stayed broken behind a passing test.
  it.each([
    ["null", null],
    ["undefined", undefined],
    ["an empty string", ""],
    ["zero", 0],
    ["false", false],
  ])("shows a panel when the thrown value was %s", (_name, thrown) => {
    vi.spyOn(console, "error").mockImplementation(() => {})
    const Throws = throwing(thrown)

    const { container } = render(
      <ErrorBoundary>
        <Throws />
      </ErrorBoundary>,
    )

    expect(screen.getByRole("alert")).toBeInTheDocument()
    expect(container).not.toBeEmptyDOMElement()
  })
})
