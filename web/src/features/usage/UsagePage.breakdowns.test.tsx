import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { UsagePage } from "@/features/usage/UsagePage"
import { API_ROOT } from "@/shared/api/client"
import { organizationMember } from "@/tests/fixtures"
import { mockApi, renderPage, summary } from "@/tests/usage"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("UsagePage breakdowns", () => {
  it("lists spend by model with a reconciling 'other' fold row", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)

    expect(await screen.findByText("gpt-5.6")).toBeInTheDocument()
    expect(screen.getByText("claude-sonnet-5")).toBeInTheDocument()
    // The null-key fold row renders as an "Other" summary, not a blank row.
    expect(screen.getByText(/Other \(14,000 req\)/)).toBeInTheDocument()
  })

  it("drills into the Activity log filtered on the clicked model", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)

    const row = (await screen.findByText("gpt-5.6")).closest("tr")!
    await user.click(row)

    const loc =
      screen.getByRole("status", { name: "Current location" }).textContent ?? ""
    expect(loc.startsWith("/activity")).toBe(true)
    expect(loc).toContain("model=gpt-5.6")
  })

  it("keeps an active user filter when drilling into a model", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    // Filter by a user, then drill into a model row. The user constraint must
    // survive the navigation, not be dropped in favor of only the clicked model.
    const userInput = screen.getByRole("combobox", { name: "User" })
    await user.click(userInput)
    await user.type(userInput, "alice")
    await user.click(await screen.findByRole("option", { name: /alice/ }))

    const row = (await screen.findByText("gpt-5.6")).closest("tr")!
    await user.click(row)

    const loc =
      screen.getByRole("status", { name: "Current location" }).textContent ?? ""
    expect(loc.startsWith("/activity")).toBe(true)
    expect(loc).toContain("model=gpt-5.6")
    expect(loc).toContain("user_id=alice")
  })

  it("keeps an active model filter when drilling into a user", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    // Filter by a model, then drill into a user row (on the User breakdown
    // tab). The model constraint must survive the navigation, not be dropped
    // in favor of only the clicked user.
    const modelInput = screen.getByRole("combobox", { name: "Model" })
    await user.click(modelInput)
    await user.type(modelInput, "gpt")
    await user.click(await screen.findByRole("option", { name: /gpt-5.6/ }))
    // The picker stays open on the remaining models (it takes several), so dismiss
    // it before reaching the page behind the overlay.
    await user.keyboard("{Escape}")

    await user.click(screen.getByRole("button", { name: "User" }))
    // The row reads as the name, not the billing id; the id is still what the
    // drill-down filters on.
    const row = (await screen.findByText("Alice")).closest("tr")!
    await user.click(row)

    const loc =
      screen.getByRole("status", { name: "Current location" }).textContent ?? ""
    expect(loc.startsWith("/activity")).toBe(true)
    expect(loc).toContain("user_id=alice")
    expect(loc).toContain("model=gpt-5.6")
  })

  it("names the person in the user breakdown instead of their billing id", async () => {
    const user = userEvent.setup()
    mockApi(
      summary({
        by_user: [
          {
            key: "81e24d08-7d1e-4287-a074-54aa57d9debc",
            label: "Alice Example",
            cost: 900.5,
            tokens: 8_000_000,
            requests: 50_000,
            is_other: false,
          },
        ],
      }),
    )
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    await user.click(screen.getByRole("button", { name: "User" }))
    const cell = await screen.findByText("Alice Example")
    // The id is still there to hover, because two people can share a name.
    expect(cell).toHaveAttribute(
      "title",
      "81e24d08-7d1e-4287-a074-54aa57d9debc",
    )
    expect(
      screen.queryByText("81e24d08-7d1e-4287-a074-54aa57d9debc"),
    ).not.toBeInTheDocument()
  })

  it("prefers the organization roster's name to the alias the log carries", async () => {
    const user = userEvent.setup()
    mockApi(summary(), {
      "/organizations/me/members": {
        data: [
          organizationMember({
            attribution_user_id: "alice",
            full_name: "Alice Example",
          }),
        ],
        total: 1,
      },
    })
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    await user.click(screen.getByRole("button", { name: "User" }))
    expect(await screen.findByText("Alice Example")).toBeInTheDocument()
    // "Alice" is the alias the summary shipped; the roster outranks it.
    expect(screen.queryByText("Alice")).not.toBeInTheDocument()
  })

  it("leaves a dimension that is already its own name alone", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)

    // A model carries no server label, so the cell is the key itself and gains
    // no title to hover.
    expect(await screen.findByText("gpt-5.6")).not.toHaveAttribute("title")
  })

  it("shows the session breakdown by default, labelling unlabelled gateway traffic", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)

    // Session is the default secondary dimension: it is what names the work
    // behind a bill for agent traffic.
    expect(await screen.findByText("project:otari")).toBeInTheDocument()
    expect(screen.getByText("Spend by session")).toBeInTheDocument()
    expect(screen.getByText("project:docs")).toBeInTheDocument()
    // Gateway rows carry no label. That is a real group, not the "other" fold,
    // so it must not read as unknown/missing data.
    expect(screen.getByText("(no session)")).toBeInTheDocument()
  })

  it("marks the active dimension button as pressed", async () => {
    // The picker's selected state cannot ride on the button variant alone: to
    // assistive tech that is four identically-named buttons with no indication of
    // which dimension the table below is showing.
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("project:otari")

    expect(screen.getByRole("button", { name: "Session" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
    expect(screen.getByRole("button", { name: "Provider" })).toHaveAttribute(
      "aria-pressed",
      "false",
    )

    await user.click(screen.getByRole("button", { name: "Provider" }))
    expect(screen.getByRole("button", { name: "Provider" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
    expect(screen.getByRole("button", { name: "Session" })).toHaveAttribute(
      "aria-pressed",
      "false",
    )
  })

  it("asks the summary endpoint only for the breakdowns the page renders", async () => {
    // Each breakdown is its own GROUP BY over the window. The page renders model,
    // user, and the four picker dimensions; the previous-period and timeline-context
    // reads use only totals/series, so they must opt out of all of them.
    const fetchMock = mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("project:otari")

    const summaryCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.includes(`${API_ROOT}/usage/summary`))
    const main = summaryCalls.find(
      (u) => u.includes("dimensions=model") && u.includes("dimensions=user"),
    )
    expect(main).toBeDefined()
    expect(main).toContain("dimensions=source_label")
    expect(main).toContain("dimensions=provider")
    // No table on this page breaks spend down by API key.
    expect(main).not.toContain("dimensions=api_key")
    expect(summaryCalls.some((u) => u.includes("dimensions=none"))).toBe(true)
  })

  it("switches the secondary breakdown between session, endpoint, provider, and source", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("project:otari")

    await user.click(screen.getByRole("button", { name: "Provider" }))
    expect(screen.getByText("Spend by provider")).toBeInTheDocument()
    expect(screen.getByText("anthropic")).toBeInTheDocument()
    expect(screen.queryByText("project:otari")).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Endpoint" }))
    expect(screen.getByText("/v1/chat/completions")).toBeInTheDocument()

    // by_source is computed and shipped by the server; it now has a home in the UI.
    await user.click(screen.getByRole("button", { name: "Source" }))
    expect(screen.getByText("claude_code")).toBeInTheDocument()
  })

  it("drills into the Activity log scoped to the clicked session", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)

    const row = (await screen.findByText("project:otari")).closest("tr")!
    await user.click(row)

    const loc =
      screen.getByRole("status", { name: "Current location" }).textContent ?? ""
    expect(loc.startsWith("/activity")).toBe(true)
    expect(loc).toContain("source_label=project%3Aotari")
  })

  it("drills into the Activity log scoped to the clicked provider", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("project:otari")

    await user.click(screen.getByRole("button", { name: "Provider" }))
    const row = screen.getByText("anthropic").closest("tr")!
    await user.click(row)

    const loc =
      screen.getByRole("status", { name: "Current location" }).textContent ?? ""
    expect(loc).toContain("provider=anthropic")
  })

  it("does not drill on the unlabelled-session row, which has no id to filter on", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)

    const row = (await screen.findByText("(no session)")).closest("tr")!
    await user.click(row)

    // Still on the Usage page: a null key cannot scope the request log.
    expect(
      screen.queryByRole("status", { name: "Current location" }),
    ).not.toBeInTheDocument()
  })
})

describe("UsagePage gateway-run tools", () => {
  it("hides the tools card while the window has no gateway-run tool calls", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    // A gateway that runs no tools should not be shown an empty table asking to be
    // explained, which is why the card is conditional rather than always present.
    expect(screen.queryByText("Gateway-run tools")).not.toBeInTheDocument()
  })

  it("shows calls, failures, and spend per tool", async () => {
    mockApi(
      summary({
        by_tool: [
          {
            tool: "web_search",
            calls: 249,
            errors: 13,
            requests: 105,
            cost: 2.49,
          },
          {
            tool: "web_fetch",
            calls: 86,
            errors: 7,
            requests: 42,
            cost: 0.43,
          },
          {
            tool: "code_execution",
            calls: 65,
            errors: 6,
            requests: 28,
            cost: 0,
          },
        ],
      }),
    )
    renderPage(<UsagePage />)

    await screen.findByText("Gateway-run tools")
    const row = screen.getByText("web search").closest("tr")!
    // Calls count tool calls, not requests: one request can search several times.
    expect(within(row).getByText("249")).toBeInTheDocument()
    expect(within(row).getByText("13")).toBeInTheDocument()
    expect(within(row).getByText("105")).toBeInTheDocument()
    expect(within(row).getByText("$2.49")).toBeInTheDocument()

    const fetchRow = screen.getByText("web fetch").closest("tr")!
    expect(within(fetchRow).getByText("86")).toBeInTheDocument()
    expect(within(fetchRow).getByText("7")).toBeInTheDocument()
    expect(within(fetchRow).getByText("42")).toBeInTheDocument()
    expect(within(fetchRow).getByText("$0.43")).toBeInTheDocument()
  })

  it("drills into the Activity log filtered on the clicked tool", async () => {
    const user = userEvent.setup()
    mockApi(
      summary({
        by_tool: [
          { tool: "web_search", calls: 12, errors: 0, requests: 7, cost: 0.12 },
        ],
      }),
    )
    renderPage(<UsagePage />)

    const row = (await screen.findByText("web search")).closest("tr")!
    await user.click(row)

    const loc =
      screen.getByRole("status", { name: "Current location" }).textContent ?? ""
    expect(loc.startsWith("/activity")).toBe(true)
    expect(loc).toContain("tool=web_search")
  })
})
