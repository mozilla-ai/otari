import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import type { ToolCall } from "./helpers/playgroundTypes"
import { formatToolArguments, ToolCalls } from "./ToolCalls"

const done: ToolCall = {
  id: "c1",
  name: "web_search",
  arguments: '{"query":"otari"}',
  result: "three hits",
  isError: false,
  isTruncated: false,
}

describe("ToolCalls", () => {
  it("shows what ran and opens to its arguments and result", async () => {
    const user = userEvent.setup()
    render(<ToolCalls calls={[done]} isStreaming={false} />)

    const trigger = screen.getByRole("button", { name: /web_search/ })
    expect(trigger).toHaveTextContent("Done")
    expect(trigger).toHaveAttribute("aria-expanded", "false")

    await user.click(trigger)
    expect(await screen.findByText("three hits")).toBeVisible()
    expect(screen.getByText(/"query": "otari"/)).toBeVisible()
  })

  it("says a call is running while the reply streams, and unanswered once it stops", () => {
    const pending: ToolCall = { ...done, result: undefined }
    const { rerender } = render(<ToolCalls calls={[pending]} isStreaming />)
    expect(screen.getByRole("button")).toHaveTextContent("Running")

    rerender(<ToolCalls calls={[pending]} isStreaming={false} />)
    expect(screen.getByRole("button")).toHaveTextContent("No result")
  })

  it("flags a failed call and a shortened result", async () => {
    const user = userEvent.setup()
    render(
      <ToolCalls
        calls={[{ ...done, isError: true, isTruncated: true }]}
        isStreaming={false}
      />,
    )
    const trigger = screen.getByRole("button", { name: /web_search/ })
    expect(trigger).toHaveTextContent("Failed")
    await user.click(trigger)
    expect(
      await screen.findByText("Result shortened for display."),
    ).toBeVisible()
  })

  it("renders nothing without calls", () => {
    const { container } = render(<ToolCalls calls={[]} isStreaming={false} />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe("formatToolArguments", () => {
  it("indents JSON, keeps anything else verbatim, and drops an empty string", () => {
    expect(formatToolArguments('{"a":1}')).toBe('{\n  "a": 1\n}')
    expect(formatToolArguments('{"a":')).toBe('{"a":')
    expect(formatToolArguments("  ")).toBe("")
  })
})
