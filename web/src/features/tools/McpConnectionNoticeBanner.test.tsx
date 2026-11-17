import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it } from "vitest"

import { McpConnectionNoticeBanner } from "@/features/tools/McpConnectionNoticeBanner"
import {
  forgetMcpConnectionNotice,
  readMcpConnectionNotice,
  rememberMcpConnectionNotice,
} from "@/features/tools/mcpConnectionNotice"

describe("McpConnectionNoticeBanner", () => {
  afterEach(() => forgetMcpConnectionNotice())

  it("says a sign-in that did not finish once, and can be dismissed", async () => {
    rememberMcpConnectionNotice({
      tone: "info",
      message: "Sign-in canceled. Nothing was connected.",
    })
    const user = userEvent.setup()
    render(<McpConnectionNoticeBanner />)

    expect(screen.getByRole("status")).toHaveTextContent("Sign-in canceled.")
    // Said once: the next page that mounts it finds nothing.
    expect(readMcpConnectionNotice()).toBeUndefined()
    await user.click(screen.getByRole("button", { name: "Dismiss" }))
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })

  it("takes a notice set while it shows, as a popup sign-in reports one", () => {
    render(<McpConnectionNoticeBanner />)
    expect(screen.queryByRole("status")).not.toBeInTheDocument()

    act(() =>
      rememberMcpConnectionNotice({
        tone: "warning",
        message: "The sign-in did not finish.",
      }),
    )

    expect(screen.getByRole("status")).toHaveTextContent(
      "The sign-in did not finish.",
    )
    expect(readMcpConnectionNotice()).toBeUndefined()
  })

  it("renders nothing when no sign-in came back", () => {
    const { container } = render(<McpConnectionNoticeBanner />)
    expect(container).toBeEmptyDOMElement()
  })
})
