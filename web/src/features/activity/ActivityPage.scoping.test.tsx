import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { API_ROOT } from "@/shared/api/client"
import { entry, jsonResponse, mockApi, renderPage } from "@/tests/activity"
import { organizationMember } from "@/tests/fixtures"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage table-scan avoidance", () => {
  it("never reads the whole users or api_keys table", async () => {
    // Both listings are fetched by paging every row (see fetchAllUsers /
    // fetchAllKeys), so a deployment with many users or keys paid a sequential
    // multi-megabyte load on every visit here, just to name filter options and
    // label a page of rows. Both now come off the summary breakdown and the
    // usage row itself. This asserts the request is gone, not merely smaller.
    const { calls } = mockApi({
      rows: [entry({ api_key_id: "key-1", api_key_name: "ci-bot" })],
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    await screen.findByText("gpt-4o")
    const requested = calls.map((c) => c.url)
    expect(requested.some((url) => url.includes(`${API_ROOT}/users`))).toBe(
      false,
    )
    expect(requested.some((url) => url.includes(`${API_ROOT}/keys`))).toBe(
      false,
    )
  })

  it("labels an API key column from the row, not a client-side lookup", async () => {
    const { calls } = mockApi({
      rows: [entry({ api_key_id: "key-1", api_key_name: "ci-bot" })],
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    expect(await screen.findByText("ci-bot")).toBeInTheDocument()
    expect(
      calls.map((c) => c.url).some((url) => url.includes(`${API_ROOT}/keys`)),
    ).toBe(false)
  })

  it("falls back to a short id when the row carries no key name", async () => {
    // The label is null whenever the key was deleted or never named, so the
    // column must not render an empty cell for a row that does have a key.
    mockApi({
      rows: [entry({ api_key_id: "abcdef123456", api_key_name: null })],
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    expect(await screen.findByText("abcdef12…")).toBeInTheDocument()
  })
})

describe("ActivityPage user naming", () => {
  it("names the user from the row's alias rather than showing the billing id", async () => {
    mockApi({
      rows: [
        entry({
          user_id: "81e24d08-7d1e-4287-a074-54aa57d9debc",
          user_alias: "Alice Example",
        }),
      ],
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    expect(await screen.findByText("Alice Example")).toBeInTheDocument()
    expect(
      screen.queryByText("81e24d08-7d1e-4287-a074-54aa57d9debc"),
    ).not.toBeInTheDocument()
  })

  it("prefers the organization roster to the alias the row carries", async () => {
    mockApi({
      rows: [
        entry({
          user_id: "81e24d08-7d1e-4287-a074-54aa57d9debc",
          user_alias: "svc-alice",
        }),
      ],
      members: [
        organizationMember({
          attribution_user_id: "81e24d08-7d1e-4287-a074-54aa57d9debc",
          full_name: "Alice Example",
        }),
      ],
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    expect(await screen.findByText("Alice Example")).toBeInTheDocument()
    expect(screen.queryByText("svc-alice")).not.toBeInTheDocument()
  })

  it("leaves an id an operator chose as its own name", async () => {
    // `ci-bot` is both the id and the alias, so naming it must not print it
    // twice or replace it with a shortened form.
    mockApi({
      rows: [entry({ user_id: "ci-bot", user_alias: "ci-bot" })],
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    expect(await screen.findByText("ci-bot")).toBeInTheDocument()
  })

  it("keeps the raw id copyable in the detail drawer", async () => {
    const user = userEvent.setup()
    mockApi({
      rows: [
        entry({
          user_id: "81e24d08-7d1e-4287-a074-54aa57d9debc",
          user_alias: "Alice Example",
        }),
      ],
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    await user.click(row)

    // The one place an operator goes for the raw id, so naming the person must
    // not take it away: the copy control still yields the id.
    const detail = row.nextElementSibling as HTMLElement
    expect(within(detail).getByText("Alice Example")).toBeInTheDocument()
    await user.click(
      within(detail).getByRole("button", { name: "Copy user id" }),
    )
    expect(await navigator.clipboard.readText()).toBe(
      "81e24d08-7d1e-4287-a074-54aa57d9debc",
    )
  })
})

describe("ActivityPage suggestion scoping", () => {
  it("keeps the user filter on the model typeahead but not on the user picker", async () => {
    // The two pickers want opposite windows. The model typeahead must stay
    // narrowed by the active user, or it offers models that user never called
    // and picking one returns an empty table. The user picker must drop it, or
    // it can only ever offer the user already selected.
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(
      <ActivityPage />,
      "/activity?model=gpt-4o&user_id=alice&range=24h",
    )

    // See the drill-down cases above: the label paints from the URL, so the wait
    // has to be on the summaries this assertion actually reads.
    const summariesSoFar = () =>
      calls
        .map((c) => c.url)
        .filter((url) => url.includes(`${API_ROOT}/usage/summary`))
    await waitFor(() =>
      expect(
        summariesSoFar().some((url) => url.includes("dimensions=user")),
      ).toBe(true),
    )
    const summaries = summariesSoFar()

    const modelQuery = summaries.find((url) => url.includes("dimensions=model"))
    expect(modelQuery, "model typeahead summary").toBeDefined()
    expect(modelQuery).toContain("user_id=alice")

    const entityQuery = summaries.find((url) => url.includes("dimensions=user"))
    expect(entityQuery, "user/key picker summary").toBeDefined()
    expect(entityQuery).not.toContain("user_id=alice")
    expect(entityQuery).toContain("model=gpt-4o")
  })
})

describe("ActivityPage for a tenant who does not operate the deployment", () => {
  it("reads the organization-scoped routes rather than the deployment-wide ones", async () => {
    const { calls } = mockApi({ rows: [entry()], deploymentOperator: false })
    renderPage(<ActivityPage />)

    await screen.findByText("gpt-4o")
    const reads = calls.filter(
      (c) => c.method === "GET" && c.url.includes("/usage"),
    )
    expect(reads).not.toHaveLength(0)
    // Every one of them, not merely one: a scope applied to the list and
    // forgotten on the count or the summary would put another tenant's totals
    // beside this tenant's rows (otari#837).
    for (const call of reads) {
      expect(call.url).toContain(`${API_ROOT}/organizations/me/usage`)
    }
  })

  it("does not poll the in-flight strip, which stays deployment-wide", async () => {
    // Its registry entries carry no workspace, so there is nothing in them to
    // scope. Polling it would be a 403 every few seconds.
    const { calls } = mockApi({ rows: [entry()], deploymentOperator: false })
    renderPage(<ActivityPage />)

    await screen.findByText("gpt-4o")
    expect(calls.some((c) => c.url.includes("/usage/in-flight"))).toBe(false)
  })

  it("offers no row selection, because the bulk writes are not theirs", async () => {
    // Deleting and repricing usage are deployment-wide. The bulk bar hangs off
    // the selection, so withholding the selection withholds both.
    const { calls } = mockApi({
      rows: [entry({ counts_toward_budget: false, source: "claude_code" })],
      deploymentOperator: false,
    })
    renderPage(<ActivityPage />)

    await screen.findByText("gpt-4o")
    expect(calls).not.toHaveLength(0)
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument()
  })

  it("still offers the selection to an operator, so the case above is not vacuous", async () => {
    mockApi({
      rows: [entry({ counts_toward_budget: false, source: "claude_code" })],
      deploymentOperator: true,
    })
    renderPage(<ActivityPage />)

    await screen.findByText("gpt-4o")
    await waitFor(() =>
      expect(screen.queryAllByRole("checkbox")).not.toHaveLength(0),
    )
  })
})

describe("ActivityPage when the organization context fails", () => {
  it("still asks for usage, and reports the refusal rather than painting an empty log", async () => {
    // The usage hooks wait on `GET /api/v1/organizations/me` to learn which surface
    // this caller may read. An errored context must not read as "keep waiting":
    // that issues no request at all, and the page then states, with no banner,
    // that a gateway serving traffic has none (otari#837).
    const calls: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      calls.push(url)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse({ detail: "no active membership" }, 404)
      }
      if (url.includes("/usage/in-flight")) {
        return jsonResponse({ requests: [], total: 0 })
      }
      if (url.includes("/usage")) {
        return jsonResponse({ detail: "context is gone" }, 403)
      }
      return jsonResponse([])
    })
    renderPage(<ActivityPage />)

    // Falls back to the narrower surface, which is the safe direction: an
    // operator reading their own organization understates, where the reverse
    // would be a cross-tenant read.
    await waitFor(() =>
      expect(
        calls.some((url) => url.includes(`${API_ROOT}/organizations/me/usage`)),
      ).toBe(true),
    )
    expect(calls.some((url) => url.startsWith(`${API_ROOT}/usage`))).toBe(false)
    // And the refusal reaches the operator instead of an empty table.
    expect(await screen.findByText(/context is gone/)).toBeInTheDocument()
  })
})
