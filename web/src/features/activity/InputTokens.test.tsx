import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { entry } from "@/tests/activity"
import { InputTokens } from "./InputTokens"

describe("InputTokens", () => {
  it("counts the cache inside the input and names the split in its title", () => {
    render(
      <InputTokens
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

    expect(screen.getByText("1,000")).toHaveAttribute(
      "title",
      "Fresh input 300, cache read 600, cache write 100",
    )
    // A title reaches a pointer only, so the split is read out as well.
    expect(
      screen.getByText(", Fresh input 300, cache read 600, cache write 100"),
    ).toHaveClass("sr-only")
  })

  it("adds no title when the cache took no part", () => {
    const { container } = render(
      <InputTokens
        entry={entry({
          billing_meters: { total_input_tokens: 400, completion_tokens: 100 },
        })}
      />,
    )

    expect(container).toHaveTextContent("400")
    expect(container.querySelector("[title]")).toBeNull()
  })

  it("falls back to the raw prompt, and to an em-dash, without usage", () => {
    const { rerender, container } = render(
      <InputTokens
        entry={entry({
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
          billing_meters: null,
        })}
      />,
    )
    expect(container).toHaveTextContent("0")

    rerender(
      <InputTokens
        entry={entry({
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
          billing_meters: null,
        })}
      />,
    )
    expect(container).toHaveTextContent("—")
  })
})
