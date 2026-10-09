import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"
import { TraceDetailPanel } from "@/features/traces/TraceDetailPanel"
import { traceDetail } from "@/tests/traceFixtures"

describe("TraceDetailPanel", () => {
  it("shows the session's turn, its LLM calls and the tool the agent ran", () => {
    render(<TraceDetailPanel detail={traceDetail()} />)

    const turn = screen.getByRole("region", { name: "Turn 1" })
    expect(within(turn).getByText("Completed")).toBeInTheDocument()
    expect(within(turn).getAllByText("gpt-5")).toHaveLength(2)
    expect(within(turn).getByText("Bash")).toBeInTheDocument()
    expect(within(turn).getByText("Recovered")).toBeInTheDocument()
  })

  it("says a client tool's duration was inferred", async () => {
    render(<TraceDetailPanel detail={traceDetail()} />)

    await userEvent.click(screen.getByRole("button", { name: /Bash/ }))

    expect(screen.getByText(/inferred/)).toBeInTheDocument()
    expect(screen.getByText("The agent")).toBeInTheDocument()
  })

  it("lists every span in order in the log view", async () => {
    render(<TraceDetailPanel detail={traceDetail()} />)

    await userEvent.click(screen.getByRole("radio", { name: "Log view" }))

    const log = screen.getByRole("list", { name: "Spans in order" })
    expect(within(log).getAllByRole("button")).toHaveLength(5)
  })

  it("collapses a request to hide what ran inside it", async () => {
    render(<TraceDetailPanel detail={traceDetail()} />)
    const turn = screen.getByRole("region", { name: "Turn 1" })
    expect(within(turn).getByText("Bash")).toBeInTheDocument()

    const [collapse] = within(turn).getAllByRole("button", {
      name: /^Collapse/,
    })
    await userEvent.click(collapse)

    expect(within(turn).queryByText("Bash")).not.toBeInTheDocument()
  })
})
