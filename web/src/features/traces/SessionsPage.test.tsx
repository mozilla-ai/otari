import { useLocation } from "@tanstack/react-router"
import { act, screen, waitFor, within } from "@testing-library/react"
import userEvent, { type UserEvent } from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { ScopedBudget } from "@/client"
import { SessionsPage } from "@/features/traces/SessionsPage"
import { jsonResponse, mockApi, renderPage } from "@/tests/activity"
import { scopedBudget } from "@/tests/fixtures"
import {
  gateSummary,
  traceDetail,
  traceSpan,
  traceSummary,
} from "@/tests/traceFixtures"

// The span a gate's check was recorded as, under the tool call it checked.
function gateCheck(gateId: string) {
  return traceSpan({
    span_id: "gate-1",
    parent_span_id: "call-toolu_1",
    kind: "guardrail",
    origin: "otlp",
    outcome: "blocked",
    attributes: {
      "otari.guardrail.profile": gateId,
      "otari.gate.id": gateId,
      "otari.gate.outcome": "fail",
      "otari.gate.blocked": true,
    },
  })
}

afterEach(() => {
  vi.restoreAllMocks()
})

// Answers one route with a 500 and leaves every other read to `mockApi`. The
// refusals sent are returned, so a test can tell when one has been read.
function refuse(
  mock: ReturnType<typeof mockApi>["mock"],
  route: RegExp,
  detail: string,
) {
  const answer = mock.getMockImplementation() as typeof fetch
  const refusals: Response[] = []
  mock.mockImplementation(async (input, init) => {
    if (!route.test(String(input))) return answer(input, init)
    const refusal = jsonResponse({ detail }, 500)
    refusals.push(refusal)
    return refusal
  })
  return refusals
}

// The query string of every read of the session list, in the order sent.
const listReads = (calls: ReturnType<typeof mockApi>["calls"]) =>
  calls
    .filter((call) => /\/traces\?/.test(call.url))
    .map((call) => new URL(call.url, "http://localhost").searchParams)

// The key's own ceiling over every provider, the one the panel reports.
const keyCeiling = (overrides: Partial<ScopedBudget> = {}) =>
  scopedBudget({
    scope_type: "api_token",
    scope_id: "key-1",
    provider_key_id: null,
    max_budget: 10,
    current_spend: 0,
    ...overrides,
  })

function SpanProbe() {
  const { searchStr } = useLocation()
  return (
    <output aria-label="Span in the URL">
      {new URLSearchParams(searchStr).get("span") ?? ""}
    </output>
  )
}

async function openFirstSession(user: Pick<UserEvent, "click"> = userEvent) {
  const sessions = await screen.findByRole("grid", { name: "Sessions" })
  await user.click(
    await within(sessions).findByRole("row", { name: /claude-code/ }),
  )
  return screen.findByRole("complementary", { name: "Trace claude-code" })
}

describe("SessionsPage", () => {
  it("opens on the span a link names, and the URL follows the span picked", async () => {
    const user = userEvent.setup()
    const summary = traceSummary()
    const detail = traceDetail()
    mockApi({
      // Off the list, as a link from Activity to an older session is.
      traces: [],
      traceDetail: {
        ...detail,
        spans: detail.spans.map((span) =>
          span.span_id === "llm-2"
            ? { ...span, model: "claude-sonnet-5" }
            : span,
        ),
      },
    })

    renderPage(
      <>
        <SessionsPage />
        <SpanProbe />
      </>,
      `/sessions?peek=${summary.trace_id}&peek_workspace=${summary.workspace_id}&span=llm-2`,
    )
    const peek = await screen.findByRole("complementary", {
      name: "Trace claude-code",
    })

    expect(
      await within(peek).findByRole("heading", { name: "claude-sonnet-5" }),
    ).toBeInTheDocument()
    expect(
      within(peek).getByRole("button", { current: true }),
    ).toHaveAccessibleName(/claude-sonnet-5/)

    await user.click(within(peek).getByRole("button", { name: /Bash/ }))

    expect(
      screen.getByRole("status", { name: "Span in the URL" }),
    ).toHaveTextContent("call-toolu_1")
    await user.click(within(peek).getByRole("button", { name: "Close" }))
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Span in the URL" }),
      ).toHaveTextContent(/^$/),
    )
  })

  it("lists sessions from the trace API and opens one into its spans", async () => {
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(await within(peek).findByText("Bash")).toBeInTheDocument()
    expect(within(peek).getAllByText("gpt-5").length).toBeGreaterThan(0)
  })

  it("reads an opened session in the workspace its list entry names", async () => {
    const { calls } = mockApi({
      traces: [traceSummary()],
      traceDetail: traceDetail(),
    })

    renderPage(<SessionsPage />, "/sessions")
    await openFirstSession()

    await vi.waitFor(() =>
      expect(
        calls.some((call) =>
          call.url.includes(
            `/traces/${traceSummary().trace_id}?workspace_id=${traceSummary().workspace_id}`,
          ),
        ),
      ).toBe(true),
    )
  })

  it("leads an opened session with its spend against the key's ceiling", async () => {
    const summary = traceSummary({ api_key_id: "key-1", cost: 0.5 })
    mockApi({
      traces: [summary],
      traceDetail: traceDetail({ summary }),
      ceilings: [keyCeiling({ max_budget: 10, current_spend: 6 })],
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(
      await within(peek).findByText(
        /This session cost \$0\.50\. That is 5\.0% of this key's \$10\.00 ceiling, which has \$6\.00 spent this period\./,
      ),
    ).toBeInTheDocument()
    expect(
      within(peek).getByRole("progressbar", {
        name: "This key's spend against its ceiling",
      }),
    ).toBeInTheDocument()
  })

  it("says when a key has no ceiling of its own", async () => {
    const summary = traceSummary({ api_key_id: "key-2" })
    mockApi({
      traces: [summary],
      traceDetail: traceDetail({ summary }),
      ceilings: [keyCeiling()],
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(
      await within(peek).findByText(/This key has no spend ceiling of its own/),
    ).toBeInTheDocument()
  })

  it("does not ask a tenant for the deployment's ceilings", async () => {
    const summary = traceSummary({ api_key_id: "key-1" })
    const { calls } = mockApi({
      traces: [summary],
      traceDetail: traceDetail({ summary }),
      deploymentOperator: false,
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(
      await within(peek).findByText("This session cost $0.02."),
    ).toBeInTheDocument()
    expect(calls.some((call) => call.url.includes("/scoped-budgets"))).toBe(
      false,
    )
  })

  it("shows how much of the session's and a call's prompt came from the cache", async () => {
    const detail = traceDetail()
    const spans = detail.spans.map((span) =>
      span.span_id === "llm-1"
        ? {
            ...span,
            input_tokens: 48_210,
            cache_read_tokens: 41_900,
            cache_write_tokens: 6_010,
            reasoning_tokens: 12,
          }
        : span,
    )
    mockApi({ traces: [traceSummary()], traceDetail: { ...detail, spans } })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    await userEvent.hover(
      await within(peek).findByRole("button", { name: "48,212 tokens" }),
    )
    expect(await screen.findByText("Cached")).toBeInTheDocument()
    expect(screen.getByText("86.9%")).toBeInTheDocument()
    await userEvent.click(
      within(peek).getAllByRole("button", { name: /^Generation, gpt-5/ })[0],
    )
    await userEvent.click(
      await within(peek).findByRole("button", { name: "Attributes" }),
    )
    expect(within(peek).getByText("48,210 (41,900 cached)")).toBeInTheDocument()
    expect(within(peek).getByText("Cache write tokens")).toBeInTheDocument()
    expect(within(peek).getByText("Reasoning tokens")).toBeInTheDocument()
  })

  it("opens the span a gateway decision is about", async () => {
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()
    await userEvent.click(
      await within(peek).findByRole("button", {
        name: /gpt-5 failed, a later attempt answered/,
      }),
    )

    expect(
      await within(peek).findByRole("button", { name: "Attributes" }),
    ).toBeInTheDocument()
    expect(
      within(peek).getByText("error, recovered by a fallback"),
    ).toBeInTheDocument()
  })

  it("lists the gates the agent's hook ran, and opens the check that blocked", async () => {
    const detail = traceDetail()
    mockApi({
      traces: [traceSummary()],
      traceDetail: {
        ...detail,
        spans: [...detail.spans, gateCheck("no-changelog")],
        gates: [
          gateSummary({
            gate_id: "no-changelog",
            message: "CHANGELOG.md is generated.",
            passed: 12,
            blocked: 1,
            flagged: 0,
            undecided: 0,
            span_id: "gate-1",
          }),
          gateSummary({
            gate_id: "python-needs-check",
            passed: 2,
            blocked: 0,
            flagged: 0,
            undecided: 0,
            span_id: null,
          }),
          gateSummary({ gate_id: "no-secret-reads", span_id: null }),
        ],
      },
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()
    const gatesTab = await within(peek).findByRole("button", {
      name: "Gates · 1 blocked",
    })
    await userEvent.click(gatesTab)

    expect(
      await within(peek).findByText(
        "2 gates ran 15 checks, and stopped the agent once.",
      ),
    ).toBeInTheDocument()
    const gate = within(peek).getByRole("button", { name: /^no-changelog/ })
    expect(gate).toHaveTextContent("1 blocked")
    expect(gate).toHaveTextContent("12 passed")
    expect(gate).toHaveTextContent("CHANGELOG.md is generated.")
    const unlinked = within(peek).getByRole("button", {
      name: /^python-needs-check/,
    })
    expect(unlinked).toHaveTextContent("2 passed")
    expect(unlinked).toBeDisabled()
    expect(
      within(peek).getByText("Did not run in this session"),
    ).toBeInTheDocument()
    expect(within(peek).getByText("no-secret-reads")).toBeInTheDocument()

    await userEvent.click(gate)

    await waitFor(() =>
      expect(
        within(peek).getByRole("button", { current: true }),
      ).toHaveAccessibleName(/no-changelog/),
    )
    expect(
      within(peek).getByRole("button", { name: "Attributes" }),
    ).toBeInTheDocument()
  })

  it("brings the check a gate blocked into view, inside a collapsed turn", async () => {
    const detail = traceDetail()
    mockApi({
      traces: [traceSummary()],
      traceDetail: {
        ...detail,
        spans: [...detail.spans, gateCheck("no-changelog")],
        gates: [
          gateSummary({
            gate_id: "no-changelog",
            passed: 0,
            blocked: 1,
            flagged: 0,
            undecided: 0,
            span_id: "gate-1",
          }),
        ],
      },
    })
    const scrolled = vi.spyOn(Element.prototype, "scrollIntoView")

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()
    await userEvent.click(
      await within(peek).findByRole("button", { name: "Collapse all" }),
    )
    expect(
      within(peek).queryByRole("button", { name: /^Tool, Bash/ }),
    ).not.toBeInTheDocument()
    await userEvent.click(within(peek).getByRole("button", { name: /^Gates/ }))
    await userEvent.click(
      await within(peek).findByRole("button", { name: /^no-changelog/ }),
    )

    await waitFor(() =>
      expect(
        within(peek).getByRole("button", { current: true }),
      ).toHaveAccessibleName(/no-changelog/),
    )
    const check = within(peek).getByRole("button", { current: true })
    expect(scrolled.mock.contexts).toContain(check)
    expect(
      within(peek).getByRole("button", { name: /^Tool, Bash/ }),
    ).toBeInTheDocument()
    scrolled.mockRestore()
  })

  it("marks a tool call's error result in the tree without calling it an error", async () => {
    const detail = traceDetail()
    mockApi({
      traces: [traceSummary()],
      traceDetail: {
        ...detail,
        spans: detail.spans.map((span) =>
          span.span_id === "call-toolu_1"
            ? { ...span, attributes: { "otari.tool.succeeded": false } }
            : span,
        ),
      },
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()
    const bash = await within(peek).findByRole("button", {
      name: /^Tool, Bash/,
    })

    expect(within(bash).getByText("Error result")).toBeInTheDocument()
    expect(within(bash).queryByText("Error")).not.toBeInTheDocument()
  })

  it("shows a quiet session's spend", async () => {
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(await within(peek).findByText("Spend")).toBeInTheDocument()
  })

  it("says what a gate is and what its check read when it is opened", async () => {
    const detail = traceDetail()
    mockApi({
      traces: [traceSummary()],
      traceDetail: {
        ...detail,
        spans: [
          ...detail.spans,
          traceSpan({
            span_id: "gate-1",
            parent_span_id: "call-toolu_1",
            kind: "guardrail",
            origin: "otlp",
            name: "gate:no-changelog",
            operation: "pre_tool_use.edit_target",
            outcome: "blocked",
            attributes: {
              "otari.gate.id": "no-changelog",
              "otari.gate.outcome": "fail",
              "otari.gate.blocked": true,
              "otari.policy.digest": "0123456789abcdef",
            },
            content: { "gate.evidence": "CHANGELOG.md" },
          }),
        ],
        gates: [
          gateSummary({
            gate_id: "no-changelog",
            blocked: 1,
            span_id: "gate-1",
            message: "CHANGELOG.md is generated.",
          }),
        ],
      },
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()
    await userEvent.click(
      await within(peek).findByRole("button", { name: /^Gates/ }),
    )
    await userEvent.click(
      await within(peek).findByRole("button", { name: /^no-changelog/ }),
    )

    expect(
      await within(peek).findByText("The gate stopped the agent here."),
    ).toBeInTheDocument()
    expect(
      within(peek).getByText("CHANGELOG.md is generated."),
    ).toBeInTheDocument()
    expect(
      within(peek).getByText("Before the agent edits a file"),
    ).toBeInTheDocument()
    expect(
      within(peek).getByText("Required: stops the agent when it fails"),
    ).toBeInTheDocument()
    expect(within(peek).getByText("What it checked")).toBeInTheDocument()
  })

  it("starts a tool call closed, its gates one row inside it", async () => {
    const detail = traceDetail()
    mockApi({
      traces: [traceSummary()],
      traceDetail: {
        ...detail,
        spans: [
          ...detail.spans,
          traceSpan({
            span_id: "gate-1",
            parent_span_id: "call-toolu_1",
            kind: "guardrail",
            origin: "otlp",
            outcome: "blocked",
            attributes: {
              "otari.gate.id": "no-changelog",
              "otari.gate.outcome": "fail",
              "otari.gate.blocked": true,
            },
          }),
        ],
      },
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()
    const tool = await within(peek).findByRole("button", {
      name: /^Tool, Bash/,
    })

    expect(
      within(peek).queryByRole("button", { name: /Gates · 1 blocked/ }),
    ).not.toBeInTheDocument()
    await userEvent.click(
      within(tool.closest("li") as HTMLElement).getByRole("button", {
        name: "Expand",
      }),
    )
    expect(
      await within(peek).findByRole("button", { name: /Gates · 1 blocked/ }),
    ).toBeInTheDocument()
    expect(
      within(peek).queryByRole("button", { name: /no-changelog/ }),
    ).not.toBeInTheDocument()
  })

  it("counts a gate that could not decide but held the agent as stopping it", async () => {
    mockApi({
      traces: [traceSummary()],
      traceDetail: {
        ...traceDetail(),
        gates: [gateSummary({ gate_id: "judge-review", refused: 1 })],
      },
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()
    await userEvent.click(
      await within(peek).findByRole("button", { name: "Gates · 1 blocked" }),
    )

    expect(
      await within(peek).findByText(
        "1 gate ran 1 check, and stopped the agent once.",
      ),
    ).toBeInTheDocument()
    expect(
      within(peek).getByText("1 could not decide, held"),
    ).toBeInTheDocument()
  })

  it("opens a link to a gate check with the tool call above it opened", async () => {
    const summary = traceSummary()
    const detail = traceDetail()
    mockApi({
      traces: [],
      traceDetail: {
        ...detail,
        spans: [
          ...detail.spans,
          traceSpan({
            span_id: "gate-1",
            parent_span_id: "call-toolu_1",
            kind: "guardrail",
            origin: "otlp",
            outcome: "blocked",
            attributes: {
              "otari.guardrail.profile": "no-changelog",
              "otari.gate.id": "no-changelog",
              "otari.gate.outcome": "fail",
              "otari.gate.blocked": true,
            },
          }),
        ],
      },
    })

    renderPage(
      <SessionsPage />,
      `/sessions?peek=${summary.trace_id}&peek_workspace=${summary.workspace_id}&span=gate-1`,
    )
    const peek = await screen.findByRole("complementary", {
      name: "Trace claude-code",
    })

    await waitFor(() =>
      expect(
        within(peek).getByRole("button", { current: true }),
      ).toHaveAccessibleName(/no-changelog/),
    )
  })

  it("leaves open a tool call another agent did work under", async () => {
    const detail = traceDetail()
    mockApi({
      traces: [traceSummary()],
      traceDetail: {
        ...detail,
        spans: [
          ...detail.spans,
          traceSpan({
            span_id: "sub-llm",
            parent_span_id: "call-toolu_1",
            kind: "llm",
            model: "subagent-model",
          }),
        ],
      },
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    const tool = await within(peek).findByRole("button", {
      name: /^Tool, Bash/,
    })
    expect(
      within(tool.closest("li") as HTMLElement).getByRole("button", {
        name: "Collapse",
      }),
    ).toBeInTheDocument()
  })

  it("has no Gates tab for a session no hook reported to", async () => {
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(
      await within(peek).findByRole("button", { name: "Log View" }),
    ).toBeInTheDocument()
    expect(
      within(peek).queryByRole("button", { name: /^Gates/ }),
    ).not.toBeInTheDocument()
  })

  it("says so when the session list cannot be read", async () => {
    const { mock } = mockApi()
    refuse(mock, /\/traces\?/, "Trace store unavailable")

    renderPage(<SessionsPage />, "/sessions")

    expect(
      await screen.findByText(/Trace store unavailable/),
    ).toBeInTheDocument()
  })

  it("says inside the open panel when the session's spans cannot be read", async () => {
    const { mock } = mockApi({ traces: [traceSummary()] })
    refuse(mock, /\/traces\/s-/, "Trace store unavailable")

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(await within(peek).findByRole("alert")).toHaveTextContent(
      "Trace store unavailable",
    )
    expect(
      within(peek).queryByRole("status", { name: "Loading the spans" }),
    ).not.toBeInTheDocument()
  })

  it("still reports what a session cost when the key's ceiling cannot be read", async () => {
    const summary = traceSummary({ api_key_id: "key-1", cost: 0.5 })
    const { mock } = mockApi({
      traces: [summary],
      traceDetail: traceDetail({ summary }),
    })
    const refusals = refuse(mock, /\/scoped-budgets/, "Budgets unavailable")

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    // A refused read renders as one still in flight does, so wait for the
    // refusal to be read and its result to reach the page before asserting
    // that it added nothing.
    await vi.waitFor(() => expect(refusals[0]?.bodyUsed).toBe(true))
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
      await new Promise((resolve) => setTimeout(resolve, 0))
    })

    expect(
      within(peek).getByText("This session cost $0.50."),
    ).toBeInTheDocument()
    expect(within(peek).queryByText(/ceiling/)).not.toBeInTheDocument()
    expect(within(peek).queryByRole("progressbar")).not.toBeInTheDocument()
  })

  it("tells a key with only a per-provider ceiling that it has none of its own", async () => {
    const summary = traceSummary({ api_key_id: "key-1" })
    mockApi({
      traces: [summary],
      traceDetail: traceDetail({ summary }),
      ceilings: [keyCeiling({ provider_key_id: "provider-key-1" })],
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(
      await within(peek).findByText(/This key has no spend ceiling of its own/),
    ).toBeInTheDocument()
    expect(within(peek).queryByRole("progressbar")).not.toBeInTheDocument()
  })

  it("searches sessions from the first page", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions?page=2")
    await vi.waitFor(() => expect(listReads(calls)[0]?.get("skip")).toBe("100"))
    await user.type(
      await screen.findByRole("searchbox", { name: "Search sessions" }),
      "codex",
    )

    await vi.waitFor(() =>
      expect(
        listReads(calls).some(
          (params) => params.get("q") === "codex" && params.get("skip") === "0",
        ),
      ).toBe(true),
    )
  })

  it("stops the search at the 128 characters the API accepts", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions")
    const search = await screen.findByRole("searchbox", {
      name: "Search sessions",
    })
    await user.click(search)
    await user.paste("x".repeat(150))

    expect(search).toHaveValue("x".repeat(128))
    await vi.waitFor(() =>
      expect(listReads(calls).at(-1)?.get("q")).toBe("x".repeat(128)),
    )
  })

  it("names the search when nothing matches it", async () => {
    mockApi({ traces: [] })

    renderPage(<SessionsPage />, "/sessions?q=codex")

    expect(
      await screen.findByText("No sessions in this window match “codex”."),
    ).toBeInTheDocument()
  })

  it("reads the window a link names", async () => {
    const { calls } = mockApi({ traces: [traceSummary()] })
    const start = "2026-10-01T00:00:00.000Z"
    const end = "2026-10-02T00:00:00.000Z"

    renderPage(
      <SessionsPage />,
      `/sessions?start_date=${start}&end_date=${end}`,
    )

    await vi.waitFor(() => expect(listReads(calls).length).toBeGreaterThan(0))
    const [params] = listReads(calls)
    expect(params.get("start")).toBe(start)
    expect(params.get("end")).toBe(end)
  })

  it("asks for the next page of sessions", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({
      traces: Array.from({ length: 26 }, (_, index) =>
        traceSummary({ trace_id: `s-${index}` }),
      ),
    })

    renderPage(<SessionsPage />, "/sessions?size=25")
    const next = await screen.findByRole("button", {
      name: "Next page, Sessions",
    })
    // Enabled once the count says there is more than one page.
    await waitFor(() => expect(next).toBeEnabled())
    await user.click(next)

    await vi.waitFor(() =>
      expect(
        listReads(calls).some(
          (params) =>
            params.get("skip") === "25" && params.get("limit") === "25",
        ),
      ).toBe(true),
    )
  })

  it("sorts by a column from the first page, largest first, and flips on a second press", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions?page=2")
    await vi.waitFor(() =>
      expect(listReads(calls)[0]?.get("sort")).toBe("last_activity"),
    )
    const cost = await screen.findByRole("columnheader", { name: /Total cost/ })
    await user.click(cost)

    await vi.waitFor(() =>
      expect(listReads(calls).at(-1)?.toString()).toMatch(
        /skip=0.*sort=cost&order=desc/,
      ),
    )
    expect(cost).toHaveAttribute("aria-sort", "descending")

    await user.click(cost)

    await vi.waitFor(() =>
      expect(listReads(calls).at(-1)?.get("order")).toBe("asc"),
    )
    expect(cost).toHaveAttribute("aria-sort", "ascending")
  })

  it("reads the sort a link names", async () => {
    const { calls } = mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions?sort=errors&order=asc")

    await vi.waitFor(() => expect(listReads(calls).length).toBeGreaterThan(0))
    expect(listReads(calls)[0].toString()).toMatch(/sort=errors&order=asc/)
    expect(
      await screen.findByRole("columnheader", { name: /Levels/ }),
    ).toHaveAttribute("aria-sort", "ascending")
  })

  it("asks for the default order when a link names a sort it does not offer", async () => {
    const { calls } = mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions?sort=name&order=sideways")

    await vi.waitFor(() => expect(listReads(calls).length).toBeGreaterThan(0))
    expect(listReads(calls)[0].toString()).toMatch(
      /sort=last_activity&order=desc/,
    )
  })

  it("moves between sessions with J and K and closes on Escape", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      mockApi({
        traces: [
          traceSummary(),
          traceSummary({ trace_id: "s-fedcba9876543210", harness: "codex" }),
        ],
        traceDetail: traceDetail(),
      })

      renderPage(<SessionsPage />, "/sessions")
      await openFirstSession(user)

      await user.keyboard("j")
      expect(
        await screen.findByRole("complementary", { name: "Trace codex" }),
      ).toBeInTheDocument()

      await user.keyboard("k")
      expect(
        await screen.findByRole("complementary", { name: "Trace claude-code" }),
      ).toBeInTheDocument()

      await user.keyboard("{Escape}")
      // The panel slides out before it is removed.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(250)
      })
      await waitFor(() =>
        expect(screen.queryByRole("complementary")).not.toBeInTheDocument(),
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("clears the span search on Escape and leaves the panel open", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

      renderPage(<SessionsPage />, "/sessions")
      const peek = await openFirstSession(user)
      const search = await within(peek).findByRole("searchbox", {
        name: "Search spans",
      })
      await user.type(search, "Bash")
      expect(
        within(peek).getByRole("list", { name: "Search results" }),
      ).toBeInTheDocument()

      await user.keyboard("{Escape}")
      await act(async () => {
        await vi.advanceTimersByTimeAsync(300)
      })

      expect(search).toHaveValue("")
      expect(
        within(peek).getByRole("list", { name: "Observations" }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole("complementary", { name: "Trace claude-code" }),
      ).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("hides a span's children when it is collapsed", async () => {
    const user = userEvent.setup()
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession(user)
    const tree = await within(peek).findByRole("list", { name: "Observations" })
    expect(
      within(tree).getByRole("button", { name: /^Tool, Bash/ }),
    ).toBeInTheDocument()

    // The first request is the parent of the Bash call.
    await user.click(
      within(tree).getAllByRole("button", { name: "Collapse" })[0],
    )

    expect(
      within(tree).queryByRole("button", { name: /^Tool, Bash/ }),
    ).not.toBeInTheDocument()
    expect(
      within(tree).getByRole("button", { name: "Expand" }),
    ).toHaveAttribute("aria-expanded", "false")
  })

  it("collapses a focused open span on ArrowLeft", async () => {
    const user = userEvent.setup()
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession(user)
    const tree = await within(peek).findByRole("list", { name: "Observations" })
    const [request] = within(tree).getAllByRole("button", {
      name: /^Span, Request/,
    })

    // Focused rather than clicked: a click also opens the span, and below md
    // that moves focus to the Back control.
    act(() => request.focus())
    await user.keyboard("{ArrowLeft}")

    expect(
      within(tree).queryByRole("button", { name: /^Tool, Bash/ }),
    ).not.toBeInTheDocument()
    expect(request).toHaveFocus()
  })

  it("draws the session as the timeline's first row and zooms it", async () => {
    const user = userEvent.setup()
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession(user)
    await user.click(
      await within(peek).findByRole("radio", { name: "Timeline" }),
    )

    const timeline = within(peek).getByRole("list", { name: "Timeline" })
    expect(within(timeline).getAllByRole("button")[0]).toHaveAccessibleName(
      /^Trace, claude-code,/,
    )
    const zoomOut = within(peek).getByRole("button", { name: "Zoom out" })
    expect(zoomOut).toBeDisabled()

    await user.click(within(peek).getByRole("button", { name: "Zoom in" }))

    expect(zoomOut).toBeEnabled()
  })

  it("says when no span matches the search", async () => {
    const user = userEvent.setup()
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession(user)
    await user.type(
      await within(peek).findByRole("searchbox", { name: "Search spans" }),
      "nothing-like-this",
    )

    expect(
      within(peek).getByText("No spans match this search."),
    ).toBeInTheDocument()
  })
})
