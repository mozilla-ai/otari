import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { entry } from "@/tests/activity"
import { TokenBar } from "./TokenBar"

describe("TokenBar", () => {
  it("shows the billed total and names every non-empty segment", () => {
    render(
      <TokenBar
        entry={entry({
          billing_meters: {
            total_input_tokens: 1000,
            cache_read_tokens: 600,
            cache_write_tokens: 100,
            completion_tokens: 200,
          },
        })}
      />,
    )

    // The sum of the segments, which for an additive-convention row is higher
    // than the raw total_tokens column.
    expect(screen.getByText("1,200")).toBeInTheDocument()
    expect(
      screen.getByRole("img", {
        name: "Token composition: Fresh input 300, Cache read 600, Cache write 100, Output 200",
      }),
    ).toBeInTheDocument()
  })

  it("leaves an empty segment out of the accessible name", () => {
    render(
      <TokenBar
        entry={entry({
          billing_meters: {
            total_input_tokens: 400,
            completion_tokens: 100,
          },
        })}
      />,
    )

    expect(
      screen.getByRole("img", {
        name: "Token composition: Fresh input 400, Output 100",
      }),
    ).toBeInTheDocument()
  })

  it("falls back to the raw total, with no bar, for a row carrying no usage", () => {
    render(
      <TokenBar
        entry={entry({
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
          billing_meters: null,
        })}
      />,
    )

    expect(screen.getByText("0")).toBeInTheDocument()
    expect(screen.queryByRole("img")).not.toBeInTheDocument()
  })

  it("shows an em dash where even the raw total is absent", () => {
    render(
      <TokenBar
        entry={entry({
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: null,
          billing_meters: null,
        })}
      />,
    )

    expect(screen.getByText("—")).toBeInTheDocument()
  })
})
