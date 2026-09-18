import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { API_ROOT } from "@/shared/api/client"
import {
  entry,
  jsonResponse,
  listCalls,
  mockApi,
  operatorContext,
  renderPage,
} from "@/tests/activity"
import { pickOption, selectTrigger } from "@/tests/select"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage filter serialization", () => {
  it("sends every active filter to the server, not just the chip", async () => {
    // The list, count, and timeline must all carry the active tool filter.
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?tool=web_search&range=24h")

    await screen.findByText("gpt-4o")
    const requested = calls.map((c) => c.url)
    for (const path of [
      `${API_ROOT}/usage?`,
      `${API_ROOT}/usage/count`,
      `${API_ROOT}/usage/summary`,
    ]) {
      const hit = requested.find((url) => url.includes(path))
      expect(hit, `no request to ${path}`).toBeDefined()
      expect(hit, `${path} dropped the tool filter`).toContain(
        "tool=web_search",
      )
    }
  })

  it("offers Web fetch as a named tool filter", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?tool=web_search&range=24h")
    await screen.findByText("gpt-4o")

    await pickOption(user, "Tool", "Web fetch")

    expect(selectTrigger("Tool")).toHaveTextContent("Web fetch")
    await waitFor(() =>
      expect(
        listCalls(calls).some((url) => url.includes("tool=web_fetch")),
      ).toBe(true),
    )
  })
})

describe("ActivityPage filters", () => {
  it("sends the api key filter to the API", async () => {
    const { calls } = mockApi({ rows: [entry({ api_key_id: "key-1" })] })
    renderPage(<ActivityPage />, "/activity?api_key_id=key-1")

    await screen.findByText("gpt-4o")
    expect(
      listCalls(calls).some((url) => url.includes("api_key_id=key-1")),
    ).toBe(true)
  })

  it("asks the summary endpoint only for the breakdowns it reads", async () => {
    // Two summary reads back this page: the model typeahead (by_model) and the
    // timeline histogram (series, plus by_tool so the Tool filter knows whether this
    // window has any gateway-run tool calls to offer). Each breakdown is a separate
    // GROUP BY over the window server-side, so neither may request the full set.
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />)

    await screen.findByText("gpt-4o")
    const summaryCalls = calls
      .filter((c) => c.url.includes(`${API_ROOT}/usage/summary`))
      .map((c) => c.url)
    expect(summaryCalls.length).toBeGreaterThan(0)
    expect(summaryCalls.some((url) => url.includes("dimensions=model"))).toBe(
      true,
    )
    expect(summaryCalls.some((url) => url.includes("dimensions=tool"))).toBe(
      true,
    )
    // No caller here reads a session/provider/user breakdown.
    expect(
      summaryCalls.some((url) => url.includes("dimensions=source_label")),
    ).toBe(false)
    expect(summaryCalls.every((url) => url.includes("dimensions="))).toBe(true)
  })

  it("honors a source drill-down and shows it as a clearable chip", async () => {
    // The pricing alarm links here scoped to gateway traffic. The param has no
    // select of its own, so if the page ignored it the banner's count and this
    // list would disagree, and the scoping would be invisible.
    const { calls } = mockApi({ rows: [entry({ status: "error" })] })
    renderPage(
      <ActivityPage />,
      "/activity?status=error&range=1h&source=gateway",
    )

    await screen.findByText("gpt-4o")
    expect(listCalls(calls).some((url) => url.includes("source=gateway"))).toBe(
      true,
    )

    const user = userEvent.setup()
    const chip = screen.getByRole("button", { name: "Remove Source filter" })
    await user.click(chip)
    await waitFor(() =>
      expect(listCalls(calls).at(-1)).not.toContain("source="),
    )
  })

  it("honors a session drill-down and shows it as a clearable chip", async () => {
    // The Usage page's session breakdown links here scoped to one source_label.
    // Without the filter the log would silently show every session's requests.
    const { calls } = mockApi({
      rows: [entry({ source: "claude_code", source_label: "sess-1" })],
    })
    renderPage(<ActivityPage />, "/activity?source_label=sess-1")

    await screen.findByText("gpt-4o")
    expect(
      listCalls(calls).some((url) => url.includes("source_label=sess-1")),
    ).toBe(true)

    const user = userEvent.setup()
    await user.click(screen.getByRole("button", { name: /Session/ }))
    await waitFor(() =>
      expect(listCalls(calls).at(-1)).not.toContain("source_label="),
    )
  })

  it("honors endpoint and provider drill-downs", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(
      <ActivityPage />,
      "/activity?endpoint=%2Fv1%2Fmessages&provider=anthropic",
    )

    await screen.findByText("gpt-4o")
    const urls = listCalls(calls)
    expect(urls.some((url) => url.includes("endpoint=%2Fv1%2Fmessages"))).toBe(
      true,
    )
    expect(urls.some((url) => url.includes("provider=anthropic"))).toBe(true)
  })

  it("sends the status filter to the API", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    const user = userEvent.setup()
    renderPage(<ActivityPage />)

    await screen.findByText("gpt-4o")
    await pickOption(user, "Status", "Error")

    await waitFor(() =>
      expect(listCalls(calls).at(-1)).toContain("status=error"),
    )
  })

  it("names a filter the URL invented rather than react-aria's placeholder", async () => {
    // A hand-edited or stale link can name a status no option carries. HeroUI's
    // Select answers an unmatched key with "Select an item", which would put
    // library boilerplate in the filter bar over a filter that is genuinely
    // applied; the control carries the value as its own option instead, so the
    // bar says what is actually filtering.
    mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?status=bogus")

    await waitFor(() =>
      expect(selectTrigger("Status")).toHaveTextContent("bogus"),
    )
    expect(selectTrigger("Status")).not.toHaveTextContent("Select an item")
  })

  it("sends the priced filter to the API", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    const user = userEvent.setup()
    renderPage(<ActivityPage />)

    await screen.findByText("gpt-4o")
    await pickOption(user, "Priced?", "Unpriced")

    await waitFor(() =>
      expect(listCalls(calls).at(-1)).toContain("priced=false"),
    )
  })

  it("offers the sources seen in the window and sends the picked one to the API", async () => {
    const { calls } = mockApi({
      rows: [
        entry(),
        entry({
          id: "imp",
          model: "claude-sonnet-4",
          source: "claude_code",
          counts_toward_budget: false,
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<ActivityPage />)
    await screen.findByText("gpt-4o")

    // Options come from the log itself (the summary's provenance breakdown), with
    // friendly labels for the sources the page knows about.
    await user.click(selectTrigger("Source"))
    await screen.findByRole("option", { name: "Claude Code" })
    expect(screen.getByRole("option", { name: "Gateway" })).toBeInTheDocument()

    await user.click(screen.getByRole("option", { name: "Claude Code" }))
    await waitFor(() =>
      expect(listCalls(calls).at(-1)).toContain("source=claude_code"),
    )
  })

  it("keeps a drill-down source listed even when the window holds none of its rows", async () => {
    // The select must show the filter that is actually applied, or the operator
    // sees a chip they cannot find in the picker.
    mockApi({ rows: [entry({ source: "gateway" })] })
    renderPage(<ActivityPage />, "/activity?source=codex")
    await screen.findByText("gpt-4o")

    const select = selectTrigger("Source")
    expect(select).toHaveTextContent("Codex")
    await userEvent.setup().click(select)
    expect(
      await screen.findByRole("option", { name: "Codex" }),
    ).toBeInTheDocument()
  })

  it("surfaces a timeline summary failure instead of an empty strip", async () => {
    // If the series query fails, "No activity in this range" would misread as a
    // quiet gateway; the error banner must carry the failure.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) return operatorContext()
      if (url.includes(`${API_ROOT}/usage/summary`)) {
        return jsonResponse({ detail: "summary exploded" }, 500)
      }
      if (url.includes(`${API_ROOT}/usage/count`))
        return jsonResponse({ total: 1 })
      if (url.includes(`${API_ROOT}/usage/in-flight`))
        return jsonResponse({ requests: [], total: 0 })
      if (url.includes(`${API_ROOT}/usage`)) return jsonResponse([entry()])
      return jsonResponse([])
    })
    renderPage(<ActivityPage />)
    await screen.findByText("gpt-4o")

    expect(await screen.findByText(/summary exploded/)).toBeInTheDocument()
  })

  it("hides the source picker while only one source exists", async () => {
    // Most gateways only ever see their own traffic; a provenance select with a
    // single option is noise, so it only appears once a second source shows up.
    mockApi({ rows: [entry(), entry({ id: "b" })] })
    renderPage(<ActivityPage />)
    await screen.findAllByText("gpt-4o")

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: /Source$/ }),
      ).not.toBeInTheDocument(),
    )
  })

  it("seeds filters from the drill-down query string", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(
      <ActivityPage />,
      "/activity?model=gpt-4o&user_id=alice&status=error",
    )

    // Waits on the request, not on "gpt-4o": that string is the model chip's own
    // label and paints from the URL before any row arrives, so finding it says
    // nothing about whether the list has been asked for yet.
    await waitFor(() => expect(listCalls(calls)).not.toHaveLength(0))
    const latest = listCalls(calls).at(-1)!
    expect(latest).toContain("model=gpt-4o")
    expect(latest).toContain("user_id=alice")
    expect(latest).toContain("status=error")
  })

  it("seeds a multi-value filter from a drill-down and keeps every value", async () => {
    // The analytics page drills with repeated params. Reading only the first would
    // silently show a narrower slice than the chart the operator clicked.
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(
      <ActivityPage />,
      "/activity?user_id=alice&user_id=bob&model=gpt-4o",
    )

    // See the case above on why this waits on the call rather than on the label.
    await waitFor(() => expect(listCalls(calls)).not.toHaveLength(0))
    const latest = listCalls(calls).at(-1)!
    expect(latest).toContain("user_id=alice")
    expect(latest).toContain("user_id=bob")

    // Both values are chips, each clearing only itself.
    expect(
      screen.getByRole("button", { name: "Remove User filter alice" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Remove User filter bob" }),
    ).toBeInTheDocument()
  })

  it("adds a second value to a filter from the picker", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?user_id=alice")
    await screen.findByText("gpt-4o")

    const userInput = screen.getByRole("combobox", { name: "User" })
    await user.click(userInput)
    await user.click(await screen.findByRole("option", { name: /bob/ }))

    await waitFor(() => {
      const latest = listCalls(calls).at(-1)!
      expect(latest).toContain("user_id=alice")
      expect(latest).toContain("user_id=bob")
    })
  })

  it("takes a free-text model value on Enter, since any model may appear in the log", async () => {
    // The model suggestions come from a windowed summary, so a model the log holds
    // but the breakdown folded away would be unfilterable if the picker were
    // options-only. Enter commits whatever was typed.
    const user = userEvent.setup()
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />)
    await screen.findByText("gpt-4o")

    const modelInput = screen.getByRole("combobox", { name: "Model" })
    await user.click(modelInput)
    await user.type(modelInput, "some-unlisted-model")
    await user.keyboard("{Enter}")

    await waitFor(() =>
      expect(listCalls(calls).at(-1)!).toContain("model=some-unlisted-model"),
    )
  })

  it("keeps the filter pickers behind an 'Add filter' toggle", async () => {
    mockApi({ rows: [entry()] })
    const user = userEvent.setup()
    renderPage(<ActivityPage />)
    await screen.findByText("gpt-4o")

    // The picker row is collapsed until the operator opts to add a filter. jsdom
    // does not apply Tailwind's `.hidden`, so assert on the toggled class rather
    // than computed visibility.
    const toggle = screen.getByRole("button", { name: "Add filter" })
    const region = document.getElementById(
      toggle.getAttribute("aria-controls")!,
    )!
    expect(toggle).toHaveAttribute("aria-expanded", "false")
    // classList, not className: `toContain` on the string is a substring match,
    // so "hidden" would also be satisfied by `overflow-hidden` and "flex" by
    // `flex-wrap` alone. Both are one edit away from being true here.
    expect([...region.classList]).toContain("hidden")

    await user.click(toggle)

    expect(toggle).toHaveAttribute("aria-expanded", "true")
    expect([...region.classList]).toContain("flex")
    expect([...region.classList]).not.toContain("hidden")
  })

  it("shows active filters as removable chips and clears one on ✕", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?model=gpt-4o&status=error")
    await screen.findByText("gpt-4o")

    // A chip per active entity filter (model + status); the time range is not a chip.
    // The entity filters hold sets, so their chips name the value they clear.
    expect(
      screen.getByRole("button", { name: "Remove Model filter gpt-4o" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Remove Status filter" }),
    ).toBeInTheDocument()

    // Removing the model chip drops just that filter from the query.
    await user.click(
      screen.getByRole("button", { name: "Remove Model filter gpt-4o" }),
    )
    await waitFor(() =>
      expect(
        listCalls(calls).some((url) => !url.includes("model=gpt-4o")),
      ).toBe(true),
    )
  })
})
