import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { entry, listCalls, mockApi, renderPage } from "@/tests/activity"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage routing plan", () => {
  it("names the model that served an absorbed attempt, from the rows already on the page", async () => {
    // The question a failed-over row raises is "so what served it". The serving
    // attempt is a sibling row sharing the request group, and it is normally on the
    // same page (the rows are written milliseconds apart), so no lookup is needed.
    const { calls } = mockApi({
      rows: [
        entry({
          id: "served",
          model: "gpt-4o",
          provider: "openai",
          status: "success",
          policy_name: "fast",
          selection_reason: "on_failure",
          attempt_position: 2,
          attempt_count: 2,
          request_group_id: "grp-1",
        }),
        entry({
          id: "absorbed",
          model: "deepseek",
          provider: "fireworks",
          status: "absorbed",
          status_code: 404,
          cost: null,
          policy_name: "fast",
          selection_reason: "default",
          attempt_position: 1,
          attempt_count: 2,
          request_group_id: "grp-1",
        }),
      ],
    })
    renderPage(<ActivityPage />)

    expect(
      await screen.findByText("attempt 1 of 2 failed, served by openai:gpt-4o"),
    ).toBeInTheDocument()
    expect(
      screen.getByText("served on attempt 2 of 2 (a fallback candidate)"),
    ).toBeInTheDocument()
    // The serving row was already listed, so nothing was looked up for it.
    expect(
      listCalls(calls).some((url) => url.includes("request_group_id=")),
    ).toBe(false)
  })

  it("looks the serving model up when the outcome row is not on the page", async () => {
    // Filtering to `absorbed` (how an operator investigates fallovers) hides every
    // outcome row by construction, so the answer has to be fetched.
    const { calls } = mockApi({
      rows: [
        entry({
          id: "absorbed",
          model: "deepseek",
          provider: "fireworks",
          status: "absorbed",
          policy_name: "fast",
          attempt_position: 1,
          attempt_count: 2,
          request_group_id: "grp-1",
        }),
      ],
      groupRows: [
        entry({
          id: "served",
          model: "gpt-4o",
          provider: "openai",
          status: "success",
          attempt_position: 2,
          attempt_count: 2,
          request_group_id: "grp-1",
        }),
      ],
    })
    renderPage(<ActivityPage />, "/activity?status=absorbed")

    expect(
      await screen.findByText("attempt 1 of 2 failed, served by openai:gpt-4o"),
    ).toBeInTheDocument()
    expect(
      listCalls(calls).some((url) => url.includes("request_group_id=grp-1")),
    ).toBe(true)
  })

  it("does not imply a fallback ran when the walk stopped early", async () => {
    // A non-retryable failure, a tool-loop lock-in, or a gateway-side refusal stops
    // the walk on the candidate it happened on, so the later candidates were never
    // called. "attempt 1 of 2" alone read as though the second one had been tried.
    mockApi({
      rows: [
        entry({
          status: "error",
          status_code: 400,
          policy_name: "fast",
          selection_reason: "default",
          attempt_position: 1,
          attempt_count: 2,
          request_group_id: "grp-1",
        }),
      ],
    })
    renderPage(<ActivityPage />)

    expect(
      await screen.findByText(
        "attempt 1 of 2 failed, no further candidate tried",
      ),
    ).toBeInTheDocument()
  })

  it("does not claim the untried candidates failed when the walk stopped mid-plan", async () => {
    // Attempt 1 was absorbed, attempt 2 stopped the walk on a non-retryable 400, so
    // candidates 3 and 4 were never called. The absorbed row must not say "and so
    // did the rest" while its sibling says "no further candidate tried".
    mockApi({
      rows: [
        entry({
          id: "absorbed",
          model: "deepseek",
          provider: "fireworks",
          status: "absorbed",
          status_code: 429,
          cost: null,
          policy_name: "fast",
          selection_reason: "default",
          attempt_position: 1,
          attempt_count: 4,
          request_group_id: "grp-1",
        }),
        entry({
          id: "stopped",
          status: "error",
          status_code: 400,
          cost: null,
          policy_name: "fast",
          selection_reason: "on_failure",
          attempt_position: 2,
          attempt_count: 4,
          request_group_id: "grp-1",
        }),
      ],
    })
    renderPage(<ActivityPage />)

    expect(
      await screen.findByText(
        "attempt 1 of 4 failed, and the request ended in an error",
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByText("attempt 2 of 4 failed, no further candidate tried"),
    ).toBeInTheDocument()
  })

  it("spells out the selection reason alone for a single-candidate policy", async () => {
    mockApi({
      rows: [
        entry({
          policy_name: "solo",
          attempt_position: 1,
          attempt_count: 1,
          selection_reason: "default",
        }),
      ],
    })
    renderPage(<ActivityPage />)

    expect(
      await screen.findByText("the policy's default target"),
    ).toBeInTheDocument()
  })

  it("humanizes a condition-matched selection reason", async () => {
    mockApi({
      rows: [
        entry({
          policy_name: "tiered",
          attempt_position: 1,
          attempt_count: 1,
          selection_reason: "condition:user_id,budget_remaining",
        }),
      ],
    })
    renderPage(<ActivityPage />)

    expect(
      await screen.findByText("matched on user_id, budget_remaining"),
    ).toBeInTheDocument()
  })

  it("shows the whole plan in the request detail, marking the attempt that served", async () => {
    mockApi({
      rows: [
        entry({
          id: "absorbed",
          model: "deepseek",
          provider: "fireworks",
          status: "absorbed",
          status_code: 404,
          cost: null,
          latency_ms: 264,
          error_message: "no pricing is configured for it",
          policy_name: "fast",
          selection_reason: "default",
          attempt_position: 1,
          attempt_count: 2,
          request_group_id: "grp-1",
        }),
      ],
      groupRows: [
        entry({
          id: "absorbed",
          model: "deepseek",
          provider: "fireworks",
          status: "absorbed",
          status_code: 404,
          cost: null,
          latency_ms: 264,
          policy_name: "fast",
          selection_reason: "default",
          attempt_position: 1,
          attempt_count: 2,
          request_group_id: "grp-1",
        }),
        entry({
          id: "served",
          model: "gpt-4o",
          provider: "openai",
          status: "success",
          cost: 0.0031,
          latency_ms: 1200,
          policy_name: "fast",
          selection_reason: "on_failure",
          attempt_position: 2,
          attempt_count: 2,
          request_group_id: "grp-1",
        }),
      ],
    })
    renderPage(<ActivityPage />)

    await userEvent.click(await screen.findByText("deepseek"))
    expect(
      await screen.findByText("Served by attempt 2 of 2: openai:gpt-4o"),
    ).toBeInTheDocument()
    const plan = screen.getByRole("table", { name: /routing plan/i })
    expect(within(plan).getByText("failed 404, fell back")).toBeInTheDocument()
    expect(within(plan).getByText("served the request")).toBeInTheDocument()
    expect(within(plan).getByText("this row")).toBeInTheDocument()
    expect(within(plan).getByText("$0.0031")).toBeInTheDocument()
  })
})

describe("ActivityPage gateway-run tools", () => {
  it("marks a row that ran tools and keeps its token bar", async () => {
    // A row can carry tool meters while its tokens were never metered (an unpriced
    // model still owes for the searches it ran). Keying the token split off the
    // presence of `billing_meters` rather than each key made the bar vanish here.
    mockApi({
      rows: [
        entry({
          prompt_tokens: 1200,
          completion_tokens: 300,
          total_tokens: 1500,
          billing_meters: {
            tools: { web_search: { billed: 3, errors: 1, unit_rate: 0.01 } },
          },
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    // 3 billed + 1 failed = 4 calls, and the detail is on the accessible name.
    const pill = within(row).getByLabelText(/Gateway tools/)
    expect(pill).toHaveTextContent("4 tools")
    expect(pill).toHaveAccessibleName("Gateway tools: web search ×3, 1 failed")
    // The bar still renders from the raw columns.
    expect(
      within(row).getByRole("img", { name: /Token composition/ }),
    ).toBeInTheDocument()
  })

  it("reads each charge line by the rate it carries, and neither by the other", async () => {
    // Three shapes reach this renderer: a per-million line, a per-call line, and
    // a line from an older gateway that matches neither. The third is the one
    // worth pinning: rendered through either rate format it would print an
    // undefined rate, so it shows the cost it did record and nothing more.
    mockApi({
      rows: [
        entry({
          cost: 0.09,
          pricing_breakdown: [
            {
              meter: "web_search_calls",
              units: 3,
              unit_rate: 0.01,
              cost: 0.03,
            },
            { meter: "input", units: 20_000, rate_per_million: 3, cost: 0.06 },
            { meter: "mystery", units: 5, cost: 0.5 },
          ],
        }),
      ],
    })
    renderPage(<ActivityPage />)

    await userEvent.click(await screen.findByText("gpt-4o"))
    await screen.findByText("Billed meters")
    // Per-million for the token line, per-call for the tool line.
    expect(screen.getByText(/20,000 at \$3\.00 \/ 1M/)).toBeInTheDocument()
    expect(screen.getByText(/3 at \$0\.01 each/)).toBeInTheDocument()
    // The legacy line shows the cost it recorded, with no rate invented for it.
    expect(screen.getByText("$0.50")).toBeInTheDocument()
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument()
  })

  it.each([
    {
      id: "a non-numeric rate",
      line: { meter: "odd", units: "many", unit_rate: "flat", cost: 1 },
    },
    // Copilot's case on #606: the discriminator is present and the field it
    // implies is missing outright, so a guard keyed on the key alone narrows it
    // and the renderer quotes a rate over no units at all.
    {
      id: "no units at all",
      line: { meter: "legacy", unit_rate: "flat", cost: 1 },
    },
  ])(
    "does not read a legacy line's rate just because the key is there: $id",
    async ({ line }) => {
      // The guard checks the shape it promises rather than the discriminator, so
      // both fall to the untyped branch instead of rendering "NaN each".
      mockApi({ rows: [entry({ cost: 1, pricing_breakdown: [line] })] })
      renderPage(<ActivityPage />)

      await userEvent.click(await screen.findByText("gpt-4o"))
      const rendered = (await screen.findByText(String(line.meter))).closest(
        "div",
      )!
      expect(within(rendered).queryByText(/each|NaN/)).not.toBeInTheDocument()
    },
  )

  it("shows tool counts and cost in the request detail", async () => {
    mockApi({
      rows: [
        entry({
          cost: 0.05,
          billing_meters: {
            tools: { web_search: { billed: 3, errors: 0, unit_rate: 0.01 } },
          },
          pricing_breakdown: [
            {
              meter: "web_search_calls",
              units: 3,
              unit_rate: 0.01,
              cost: 0.03,
            },
          ],
        }),
      ],
    })
    renderPage(<ActivityPage />)

    await userEvent.click(await screen.findByText("gpt-4o"))
    expect(await screen.findByText("Tools")).toBeInTheDocument()
    expect(screen.getByText("web search ×3")).toBeInTheDocument()
    // Per-call charge lines read "N at $X each", not the token form "$X / 1M".
    expect(screen.getByText(/3 at \$0\.01 each, \$0\.03/)).toBeInTheDocument()
  })

  it.each([
    {
      name: "failed Fetch only",
      tools: { web_fetch: { billed: 0, errors: 1 } },
      expectedCost: "$0.00",
    },
    {
      name: "failed Fetch and priced Search",
      tools: {
        web_fetch: { billed: 0, errors: 1 },
        web_search: { billed: 1, errors: 0, unit_rate: 0.01 },
      },
      expectedCost: "$0.01",
    },
  ])(
    "shows the billed tool cost for $name",
    async ({ tools, expectedCost }) => {
      mockApi({ rows: [entry({ billing_meters: { tools } })] })
      renderPage(<ActivityPage />)

      await userEvent.click(await screen.findByText("gpt-4o"))
      const costField = (await screen.findByText("Tool cost")).closest("div")!
      expect(within(costField).getByText(expectedCost)).toBeInTheDocument()
      expect(screen.getByText(/web fetch, 1 failed/)).toBeInTheDocument()
      expect(screen.queryByText("unpriced")).not.toBeInTheDocument()
    },
  )

  it("does not hide an unpriced successful tool beside a priced tool", async () => {
    mockApi({
      rows: [
        entry({
          billing_meters: {
            tools: {
              web_fetch: { billed: 1, errors: 0 },
              web_search: { billed: 1, errors: 0, unit_rate: 0.01 },
            },
          },
        }),
      ],
    })
    renderPage(<ActivityPage />)

    await userEvent.click(await screen.findByText("gpt-4o"))
    expect(await screen.findByText("unpriced")).toBeInTheDocument()
  })

  it("labels an unpriced tool instead of reporting it as free", async () => {
    // A tool with no rate records units at cost 0. Rendering that as "$0.0000"
    // would read as "this is free" when it means "nobody set a price".
    mockApi({
      rows: [
        entry({
          billing_meters: { tools: { web_search: { billed: 2, errors: 0 } } },
        }),
      ],
    })
    renderPage(<ActivityPage />)

    await userEvent.click(await screen.findByText("gpt-4o"))
    expect(await screen.findByText("Tool cost")).toBeInTheDocument()
    expect(screen.getByText("unpriced")).toBeInTheDocument()
  })
})
