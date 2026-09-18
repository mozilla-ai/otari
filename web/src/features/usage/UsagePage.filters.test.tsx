import { screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { UsagePage } from "@/features/usage/UsagePage"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import { pickOption, selectTrigger } from "@/tests/select"
import { mockApi, renderPage, summary } from "@/tests/usage"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("UsagePage scope and filters", () => {
  it("scopes the workspace view to the switcher's selection", async () => {
    const fetchMock = mockApi(summary(), {
      [`${API_ROOT}/organizations/me`]: organizationContext({
        workspace_memberships: [
          {
            workspace_id: "ws-1",
            name: "Platform team",
            role: "owner",
          },
        ],
      }),
    })
    renderPage(<UsagePage />, { scoped: true })
    await screen.findByText("$1,240.50")

    expect(
      fetchMock.mock.calls.some(
        ([url]) =>
          String(url).includes(`${API_ROOT}/usage/summary`) &&
          String(url).includes("workspace_id=ws-1"),
      ),
    ).toBe(true)
    // The workspace rail takes its scope from the switcher and offers no
    // workspace picker, so the roster this page never shows is not fetched.
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes(`${API_ROOT}/workspaces`),
      ),
    ).toBe(false)
  })

  it("asks for the whole organization on the organization page, whatever the switcher selects", async () => {
    // The switcher holds a selection and the fixture's caller even operates the
    // deployment; the organization page must let neither leak in. Narrowed by
    // the switcher it would repeat the workspace page, and widened to
    // /api/v1/usage it would title every tenant's traffic as this organization's.
    const fetchMock = mockApi(summary(), {
      [`${API_ROOT}/organizations/me`]: organizationContext({
        workspace_memberships: [
          {
            workspace_id: "ws-1",
            name: "Platform team",
            role: "owner",
          },
        ],
      }),
    })
    renderPage(<UsagePage scope="organization" />, { scoped: true })
    await screen.findByText("$1,240.50")

    const reads = fetchMock.mock.calls
      .map(([url]) => String(url))
      .filter((url) => url.includes("/usage/"))
    expect(reads).not.toHaveLength(0)
    for (const url of reads) {
      expect(url).toContain(`${API_ROOT}/organizations/me/usage/`)
      expect(url).not.toContain("workspace_id=")
    }
  })

  it("narrows the organization page through its own workspace filter", async () => {
    const user = userEvent.setup()
    const fetchMock = mockApi(summary(), {
      [`${API_ROOT}/workspaces`]: {
        data: [
          {
            id: "ws-2",
            name: "Research",
            organization_id: "org-1",
            created_at: "2026-08-01T00:00:00+00:00",
            updated_at: "2026-08-01T00:00:00+00:00",
          },
        ],
        count: 1,
      },
    })
    renderPage(<UsagePage scope="organization" />)
    await screen.findByText("$1,240.50")

    // Through the disclosure a person uses. jsdom does not apply Tailwind's
    // `.hidden`, so the select is reachable without this and the case would
    // keep passing if the control became unreachable in a browser.
    await user.click(screen.getByRole("button", { name: "Add filter" }))
    await pickOption(user, "Workspace", "Research")

    const summaryCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.includes(`${API_ROOT}/organizations/me/usage/summary`))
    expect(summaryCalls.some((u) => u.includes("workspace_id=ws-2"))).toBe(true)
    // The narrowing is visible and revocable where every other filter is.
    expect(
      screen.getByRole("button", {
        name: "Remove Workspace filter Research",
      }),
    ).toBeInTheDocument()
  })

  it("filters usage by API key", async () => {
    const user = userEvent.setup()
    const fetchMock = mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    await user.click(screen.getByPlaceholderText("All keys"))
    await user.click(await screen.findByRole("option", { name: "ci-bot" }))

    const summaryCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.includes(`${API_ROOT}/usage/summary`))
    expect(summaryCalls.some((u) => u.includes("api_key_id=key-1"))).toBe(true)
  })

  it("keeps an active API key filter when drilling into a model", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    // Filter by an API key, then drill into a model row. The key constraint must
    // survive the navigation alongside the clicked model.
    await user.click(screen.getByPlaceholderText("All keys"))
    await user.click(await screen.findByRole("option", { name: "ci-bot" }))

    const row = (await screen.findByText("gpt-5.6")).closest("tr")!
    await user.click(row)

    const loc =
      screen.getByRole("status", { name: "Current location" }).textContent ?? ""
    expect(loc.startsWith("/activity")).toBe(true)
    expect(loc).toContain("model=gpt-5.6")
    expect(loc).toContain("api_key_id=key-1")
  })

  it("filters models by typeahead and commits the exact picked model", async () => {
    const fetchMock = mockApi(summary())
    const user = userEvent.setup()
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    // The model box is a typeahead sourced from the in-window models, not a
    // free-text exact-match input.
    const modelInput = screen.getByRole("combobox", { name: "Model" })
    await user.click(modelInput)
    await user.type(modelInput, "claude")
    await user.click(
      await screen.findByRole("option", { name: /claude-sonnet-5/ }),
    )

    const summaryCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.includes("/usage/summary"))
    expect(summaryCalls.at(-1)).toContain("model=claude-sonnet-5")
  })

  it("hides the source dimension while only one source exists", async () => {
    // A plain gateway: every row shares one source, so neither the breakdown
    // tab nor the group-by option should surface provenance.
    mockApi(
      summary({
        by_source: [
          {
            key: "gateway",
            cost: 1240.5,
            tokens: 12_400_000,
            requests: 84_000,
            is_other: false,
          },
        ],
      }),
    )
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    expect(
      screen.queryByRole("button", { name: "Source" }),
    ).not.toBeInTheDocument()
    await userEvent.setup().click(selectTrigger("Group by"))
    expect(
      screen.queryByRole("option", { name: "By source" }),
    ).not.toBeInTheDocument()
  })

  it("offers the source dimension once several sources exist", async () => {
    mockApi(
      summary({
        by_source: [
          {
            key: "gateway",
            cost: 900,
            tokens: 9_000_000,
            requests: 60_000,
            is_other: false,
          },
          {
            key: "claude_code",
            cost: 340.5,
            tokens: 3_400_000,
            requests: 24_000,
            is_other: false,
          },
        ],
      }),
    )
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    expect(screen.getByRole("button", { name: "Source" })).toBeInTheDocument()
    await userEvent.setup().click(selectTrigger("Group by"))
    expect(
      await screen.findByRole("option", { name: "By source" }),
    ).toBeInTheDocument()
  })

  it("keeps the filter pickers behind an 'Add filter' toggle", async () => {
    mockApi(summary())
    const user = userEvent.setup()
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    const toggle = screen.getByRole("button", { name: "Add filter" })
    const region = document.getElementById(
      toggle.getAttribute("aria-controls")!,
    )!
    // jsdom does not apply Tailwind's `.hidden`, so assert on the class the toggle
    // flips (display:none collapsed, flex expanded) rather than computed visibility.
    expect(toggle).toHaveAttribute("aria-expanded", "false")
    expect(region.className).toContain("hidden")

    await user.click(toggle)

    expect(toggle).toHaveAttribute("aria-expanded", "true")
    // classList, not className: `toContain` on the string is a substring match,
    // so "flex" is satisfied by `flex-wrap` alone and "hidden" by
    // `overflow-hidden`. Both are one edit away from being true here.
    expect([...region.classList]).toContain("flex")
    expect([...region.classList]).not.toContain("hidden")
  })

  it("surfaces an active filter as a removable chip", async () => {
    mockApi(summary())
    const user = userEvent.setup()
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    // No entity filters yet, so no chips.
    expect(
      screen.queryByRole("button", { name: /Remove .* filter/ }),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Add filter" }))
    await user.click(screen.getByPlaceholderText("All keys"))
    await user.click(await screen.findByRole("option", { name: "ci-bot" }))
    // The picker stays open on the remaining keys; dismiss it to reach the chips.
    await user.keyboard("{Escape}")

    // The picked key shows as a chip whose remove control names the value: a
    // dimension can hold several, so the label has to distinguish them.
    expect(
      await screen.findByRole("button", {
        name: "Remove API key filter ci-bot",
      }),
    ).toBeInTheDocument()
  })

  it("filters the chart on several models at once", async () => {
    const user = userEvent.setup()
    const fetchMock = mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    // A spend question is usually a comparison, so the picker accumulates values
    // and sends them as repeated params (the endpoints match any of them).
    const modelInput = screen.getByRole("combobox", { name: "Model" })
    await user.click(modelInput)
    await user.click(await screen.findByRole("option", { name: "gpt-5.6" }))
    await user.click(
      await screen.findByRole("option", { name: "claude-sonnet-5" }),
    )

    await vi.waitFor(() => {
      const last = fetchMock.mock.calls
        .map(([u]) => String(u))
        .filter((u) => u.includes(`${API_ROOT}/usage/summary`))
        .at(-1)
      expect(last).toContain("model=gpt-5.6")
      expect(last).toContain("model=claude-sonnet-5")
    })

    // Both picks carry their own chip, and removing one leaves the other applied.
    await user.keyboard("{Escape}")
    expect(
      screen.getByRole("button", { name: "Remove Model filter gpt-5.6" }),
    ).toBeInTheDocument()
    await user.click(
      screen.getByRole("button", {
        name: "Remove Model filter claude-sonnet-5",
      }),
    )

    expect(
      screen.queryByRole("button", {
        name: "Remove Model filter claude-sonnet-5",
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Remove Model filter gpt-5.6" }),
    ).toBeInTheDocument()
  })

  it("carries a whole multi-value filter into the request log", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    // A two-user comparison travels as repeated params, so the log opens on exactly
    // the traffic the chart was showing rather than on a wider or arbitrary slice.
    const userInput = screen.getByRole("combobox", { name: "User" })
    await user.click(userInput)
    await user.click(await screen.findByRole("option", { name: /alice/ }))
    await user.click(await screen.findByRole("option", { name: /bob/ }))
    await user.keyboard("{Escape}")

    const row = (await screen.findByText("gpt-5.6")).closest("tr")!
    await user.click(row)

    const loc =
      screen.getByRole("status", { name: "Current location" }).textContent ?? ""
    expect(loc).toContain("model=gpt-5.6")
    expect(loc).toContain("user_id=alice")
    expect(loc).toContain("user_id=bob")
  })
})
