import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { RoutingPage } from "@/features/routing/RoutingPage"
import { CHAIN, LEARNED, mockApi, policy, renderPage } from "@/tests/routing"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage router examples", () => {
  it("offers the examples panel only on a policy that actually uses a router", async () => {
    // Readiness is a per-policy question, so it belongs on the row like Edit does
    // rather than in a panel that is always on the page.
    mockApi([
      policy("smart", LEARNED, { is_dynamic: true }),
      policy("fast", CHAIN),
    ])
    renderPage(<RoutingPage />)

    const learnedRow = (await screen.findByText("smart")).closest("tr")!
    const plainRow = (await screen.findByText("fast")).closest("tr")!
    expect(
      within(learnedRow).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
    expect(
      within(plainRow).queryByRole("button", { name: "Examples" }),
    ).not.toBeInTheDocument()
    // Nothing about learned routing is on the page until asked for.
    expect(screen.queryByText(/Whose memory/)).not.toBeInTheDocument()
  })

  it("offers the examples panel for a config.yml policy, which cannot be edited", async () => {
    // Reading readiness is safe for a policy this page cannot change, and without it
    // a config-defined learned policy would be entirely opaque here.
    mockApi([policy("smart", LEARNED, { is_dynamic: true, source: "config" })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    expect(
      within(row).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(within(row).getByText("set in config.yml")).toBeInTheDocument()
  })

  it("names the pool and what serves when the router declines", async () => {
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      await screen.findByText(/ranks openai:gpt-5-nano, openai:gpt-5/),
    ).toBeInTheDocument()
    expect(screen.getByText(/serves whenever it declines/)).toBeInTheDocument()
    // The honest empty state: no user picked yet, so no warmth claim.
    expect(screen.getByText(/Pick a user to see how warm/)).toBeInTheDocument()
  })

  it("reports each pool's warmth for the chosen user, since memory is per user", async () => {
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))
    await user.type(
      screen.getByRole("combobox", { name: /whose memory/i }),
      "alice",
    )
    await user.keyboard("{Escape}")

    expect(await screen.findByText("6 / 20 examples")).toBeInTheDocument()
    expect(screen.getByText("WARMING UP")).toBeInTheDocument()
    // A task partition warms on its own, so it gets its own line.
    expect(screen.getByText("summaries")).toBeInTheDocument()
    expect(screen.getByText("21 / 20 examples")).toBeInTheDocument()
    expect(screen.getByText("ROUTING")).toBeInTheDocument()
  })

  it("says where examples come from instead of offering to collect them", async () => {
    // Recording examples is an API job in this release. The panel has to say so, or
    // an operator reads "0 examples" as a bug with no next step.
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      await screen.findByText(/POST \/api\/v1\/routing\/preferences\/rank/),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /teach it/i })).toBeInTheDocument()
    // No write affordance anywhere in it.
    expect(
      screen.queryByRole("button", { name: /ask all/i }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /record these scores/i }),
    ).not.toBeInTheDocument()
  })

  it("does not ask whose memory for a user-scoped policy", async () => {
    // A policy scoped to one user can only use that user's memory, so asking would
    // be a question with one answer, and a wrong answer would be accepted.
    mockApi([policy("smart", LEARNED, { is_dynamic: true, user_id: "alice" })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      screen.queryByRole("combobox", { name: /whose memory/i }),
    ).not.toBeInTheDocument()
    expect(await screen.findByText("6 / 20 examples")).toBeInTheDocument()
  })
})
