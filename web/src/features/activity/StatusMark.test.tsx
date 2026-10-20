import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { StatusMark } from "./StatusMark"

describe("StatusMark", () => {
  it("names each known status in its own casing", () => {
    render(<StatusMark status="absorbed" />)
    expect(screen.getByText("Absorbed")).toBeInTheDocument()
  })

  it("colors only a failure's text, so it is findable in a scan of rows", () => {
    // The class is the assertion because there is no accessible expression of
    // "this word is red"; it is reached through the word itself, and the
    // neighbouring statuses pin that the coloring is not applied everywhere.
    const { rerender } = render(<StatusMark status="error" />)
    expect(screen.getByText("Error")).toHaveClass("text-danger")

    rerender(<StatusMark status="success" />)
    expect(screen.getByText("Success")).not.toHaveClass("text-danger")

    rerender(<StatusMark status="absorbed" />)
    expect(screen.getByText("Absorbed")).not.toHaveClass("text-danger")
  })

  it("renders an unknown status as its own slug, not as a success", () => {
    render(<StatusMark status="cancelled_by_client" />)
    const word = screen.getByText("cancelled_by_client")
    expect(word).toHaveClass("text-subtle", "truncate")
    expect(word.parentElement).toHaveAttribute("title", "cancelled_by_client")
  })

  it("shows a skipped candidate as neither a failure nor a success", () => {
    render(<StatusMark status="skipped" />)
    const word = screen.getByText("Skipped")
    expect(word).toHaveClass("text-subtle")
  })
})
