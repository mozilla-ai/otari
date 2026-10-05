import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { entry } from "@/tests/activity"
import { RoutingCell } from "./RoutingCell"

describe("RoutingCell", () => {
  it("names the policy and where the row sits in its plan", () => {
    render(
      <RoutingCell
        entry={entry({
          policy_name: "cheap-first",
          status: "absorbed",
          attempt_position: 1,
          attempt_count: 2,
        })}
        outcome={{ servedBy: "openai:gpt-4o", servedPosition: 2 }}
      />,
    )

    expect(screen.getByText("cheap-first")).toBeInTheDocument()
    expect(
      screen.getByText("attempt 1 of 2 failed, served by openai:gpt-4o"),
    ).toBeInTheDocument()
  })

  it("says a candidate was skipped for a full rate limit, not that it failed", () => {
    render(
      <RoutingCell
        entry={entry({
          policy_name: "spill",
          status: "absorbed",
          status_code: 429,
          error_message:
            "Skipped: Rate limit 'flash-cap' exceeded: 2 requests per minute",
          attempt_position: 1,
          attempt_count: 2,
        })}
        outcome={{
          servedBy: "anthropic:claude-haiku-4-5",
          servedPosition: 2,
        }}
      />,
    )

    expect(
      screen.getByText(
        "attempt 1 of 2 skipped, rate limit 'flash-cap' full, served by anthropic:claude-haiku-4-5",
      ),
    ).toBeInTheDocument()
  })

  it("renders nothing for a request that named a plain model", () => {
    // Sparse by nature: a placeholder on every unrouted row would add noise to
    // every scan while saying nothing.
    const { container } = render(
      <RoutingCell entry={entry({ policy_name: null })} outcome={null} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it("shows the policy alone when the row has no sentence to add", () => {
    render(
      <RoutingCell
        entry={entry({
          policy_name: "cheap-first",
          attempt_position: null,
          attempt_count: null,
          selection_reason: null,
        })}
        outcome={null}
      />,
    )
    expect(screen.getByText("cheap-first")).toBeInTheDocument()
    expect(screen.queryByText(/attempt/)).not.toBeInTheDocument()
  })
})
