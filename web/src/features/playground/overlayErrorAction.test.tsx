import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { MessageBubble } from "@/features/playground/MessageBubble"

/**
 * The base default, through the bubble that mounts it. `PlaygroundPage.test.tsx`
 * replaces the module to show what a failed turn hands it; this file holds the
 * other half, that this build adds nothing under the error text.
 */
describe("the base Playground error action", () => {
  it("leaves a refused turn showing only its error text", () => {
    const { container } = render(
      <MessageBubble
        model="openai:gpt-4o"
        turn={{
          role: "assistant",
          content: "",
          error: {
            message: "Insufficient funds ($0.00) to serve this request.",
            status: 402,
            code: "insufficient_funds",
          },
        }}
      />,
    )

    const banner = screen.getByText(
      "Insufficient funds ($0.00) to serve this request.",
    )
    expect(banner).toBeInTheDocument()
    expect(container.querySelector("a")).toBeNull()
    expect(
      screen.queryByRole("button", { name: /fund/i }),
    ).not.toBeInTheDocument()
  })
})
