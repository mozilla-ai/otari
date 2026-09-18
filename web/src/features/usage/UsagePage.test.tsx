import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { UsagePage } from "@/features/usage/UsagePage"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext, seriesPoint, usageTotals } from "@/tests/fixtures"
import { pickOption } from "@/tests/select"
import { jsonResponse, mockApi, renderPage, summary } from "@/tests/usage"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("UsagePage", () => {
  it("renders totals tiles with compact currency and error rate", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)

    // The total ($1,240.50) is unique to the tile: no single breakdown row equals it.
    expect(await screen.findByText("$1,240.50")).toBeInTheDocument()
    expect(screen.getByText("84,000")).toBeInTheDocument()
    expect(screen.getByText("12.4M")).toBeInTheDocument()
    // 1764 / 84000 = 2.1% errors.
    expect(screen.getByText(/2\.1% errors/)).toBeInTheDocument()
  })

  it("does not render the CSV export action", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)

    await screen.findByText("$1,240.50")
    expect(
      screen.queryByRole("button", { name: "Export CSV" }),
    ).not.toBeInTheDocument()
  })

  it("puts share in the chart's own caption row, as an icon with no visible label", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)

    await screen.findByText("$1,240.50")
    const share = screen.getByRole("button", {
      name: "Share usage as an image",
    })
    // On the artifact it publishes, not among the page's global controls.
    expect(share.closest("figure")).not.toBeNull()
    expect(share.closest("figcaption")).not.toBeNull()
    // Icon-only. The accessible name comes from aria-label, so there is no text.
    expect(share).toHaveTextContent("")
    expect(share.querySelector("svg")).not.toBeNull()
  })

  it("offers no share affordance when the range has no data to share", async () => {
    mockApi(summary({ series: [] }))
    renderPage(<UsagePage />)

    await screen.findByText("No data in this range.")
    expect(
      screen.queryByRole("button", { name: "Share usage as an image" }),
    ).not.toBeInTheDocument()
  })

  it("shares whatever the page is filtered to, with no separate share query", async () => {
    const user = userEvent.setup()
    const fetchMock = mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    await user.click(
      screen.getByRole("button", { name: "Share usage as an image" }),
    )
    // By role: a `Dialog` fills HeroUI's trigger slot with its own title and
    // hides it, so the string is in the document twice.
    await screen.findByRole("dialog", { name: "Share this view as an image" })

    // The panel reads the page's own summary. If it ever grows a query of its
    // own, opening it would add a /v1/usage/summary call with a different
    // dimension set, and the card could then disagree with the page above it.
    const shareCalls = fetchMock.mock.calls.filter((call) =>
      String(call[0]).includes("provider_model"),
    )
    expect(shareCalls).toHaveLength(0)
  })

  it("re-queries with an hourly bucket when a sub-day preset is chosen from the timeline", async () => {
    const user = userEvent.setup()
    const fetchMock = mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")
    fetchMock.mockClear()

    await user.click(screen.getByRole("button", { name: "Last hour" }))

    await vi.waitFor(() => {
      const summaryCalls = fetchMock.mock.calls
        .map(([u]) => String(u))
        .filter((u) => u.includes(`${API_ROOT}/usage/summary`))
      // The sub-day extent buckets hourly (both the context histogram and the tiles).
      expect(summaryCalls.some((u) => u.includes("bucket=hour"))).toBe(true)
    })
  })

  it("shows the cache story as a hit rate with read/write volumes", async () => {
    const base = summary()
    mockApi(
      summary({
        // The tile reads the meter-normalized series composition (the same
        // numbers as its sparkline), not the raw totals columns.
        series: base.series.map((p, i) => ({
          ...p,
          input_tokens: i === 0 ? 4_200_000 : 6_000_000,
          cache_read_tokens: i === 0 ? 2_100_000 : 3_000_000,
          cache_write_tokens: i === 0 ? 1_200_000 : 1_500_000,
          output_tokens: 400_000,
        })),
      }),
    )
    renderPage(<UsagePage />)

    // Await a value (loads after the query resolves), not the static label.
    // 5.1M reads over 10.2M billed input tokens = a 50.0% hit rate.
    expect(await screen.findByText("50.0%")).toBeInTheDocument()
    expect(screen.getByText("Cache hit rate")).toBeInTheDocument()
    expect(screen.getByText(/5\.1M read · 2\.7M written/)).toBeInTheDocument()
  })

  it("shows no hit rate when the window carries no input-token composition", async () => {
    // The rate is series cache reads over series input tokens, so what makes it
    // uncomputable is a window with no input composition, which is what an older
    // gateway (vite dev against a stale build) returns.
    //
    // Cache reads are present on purpose: without them the tile reads "—" for
    // want of a numerator and the denominator never matters, which is how this
    // passed while naming a field it does not read.
    const noComposition = [
      seriesPoint({
        bucket_start: "2026-07-19T00:00:00Z",
        cache_read_tokens: 2_000_000,
      }),
      seriesPoint({
        bucket_start: "2026-07-20T00:00:00Z",
        cache_read_tokens: 3_100_000,
      }),
    ]
    mockApi(summary({ series: noComposition }))
    renderPage(<UsagePage />)

    await screen.findByText("$1,240.50")
    const tile = screen.getByText("Cache hit rate").closest("div")!
    expect(within(tile).getByText("—")).toBeInTheDocument()
  })

  it("computes the hit rate once the window has input tokens to divide by", async () => {
    // The other side of the branch above, so the em-dash case is pinned to the
    // missing composition rather than to anything else about the fixture.
    const withComposition = [
      seriesPoint({
        bucket_start: "2026-07-19T00:00:00Z",
        input_tokens: 4_000_000,
        cache_read_tokens: 2_000_000,
      }),
      seriesPoint({
        bucket_start: "2026-07-20T00:00:00Z",
        input_tokens: 6_000_000,
        cache_read_tokens: 3_000_000,
      }),
    ]
    mockApi(summary({ series: withComposition }))
    renderPage(<UsagePage />)

    await screen.findByText("$1,240.50")
    const tile = screen.getByText("Cache hit rate").closest("div")!
    expect(within(tile).getByText("50.0%")).toBeInTheDocument()
  })

  it("groups the chart by a dimension via the grouped series endpoint", async () => {
    const user = userEvent.setup()
    const fetchMock = mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    await pickOption(user, "Group by", "By model")

    // The stack's legend comes from the grouped response: the top group plus
    // the reconciling fold, which always reads "Other".
    expect(await screen.findByText("Other")).toBeInTheDocument()
    const calls = fetchMock.mock.calls.map(([u]) => String(u))
    expect(
      calls.some(
        (u) =>
          u.includes(`${API_ROOT}/usage/series`) &&
          u.includes("group_by=model"),
      ),
    ).toBe(true)
  })

  it("falls back to ungrouped with a notice when the gateway lacks grouped series", async () => {
    // Version skew: the dashboard ships inside the gateway, but a not-yet
    // restarted gateway (or vite dev against an older one) has no
    // /v1/usage/series. That must degrade to the ungrouped chart plus a
    // notice, not spin through retries into a bare "Not Found" banner.
    const user = userEvent.setup()
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      // The shell reads this before it paints, and the usage hooks wait for it:
      // it is what tells them whether this caller reads the deployment-wide
      // routes or the organization-scoped ones (otari#837). Answered first, and
      // on an exact match, so it cannot shadow /v1/organizations/me/usage.
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(organizationContext())
      }
      if (url.includes(`${API_ROOT}/usage/series`))
        return jsonResponse({ detail: "Not Found" }, 404)
      if (url.includes(`${API_ROOT}/usage/summary`))
        return jsonResponse(summary())
      return jsonResponse([])
    })
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    await pickOption(user, "Group by", "By model")

    expect(
      await screen.findByText(/predates grouped series/),
    ).toBeInTheDocument()
    expect(screen.queryByText("Not Found")).not.toBeInTheDocument()
    // The ungrouped single-series chart is still up (its caption renders).
    expect(screen.getByText(/peak/)).toBeInTheDocument()
  })

  it("stacks the billed token composition on the Tokens metric", async () => {
    const user = userEvent.setup()
    const base = summary()
    mockApi(
      summary({
        series: base.series.map((p) => ({
          ...p,
          input_tokens: 3_000_000,
          cache_read_tokens: 2_000_000,
          cache_write_tokens: 500_000,
          output_tokens: 400_000,
        })),
      }),
    )
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    await user.click(screen.getByRole("button", { name: "Tokens" }))

    // The four billed buckets are legended: the same encoding as the Activity
    // page's per-row token bar.
    expect(await screen.findByText("Fresh input")).toBeInTheDocument()
    expect(screen.getByText("Cache read")).toBeInTheDocument()
    expect(screen.getByText("Cache write")).toBeInTheDocument()
    expect(screen.getByText("Output")).toBeInTheDocument()
  })

  it("splits requests into succeeded and failed when the window has errors", async () => {
    const user = userEvent.setup()
    const base = summary()
    mockApi(
      summary({ series: base.series.map((p) => ({ ...p, errors: 100 })) }),
    )
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    await user.click(screen.getByRole("button", { name: "Requests" }))

    expect(await screen.findByText("Failed")).toBeInTheDocument()
    expect(screen.getByText("Succeeded")).toBeInTheDocument()
  })

  it("queries the previous period with a bounded end_date for deltas", async () => {
    const fetchMock = mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    const summaryCalls = fetchMock.mock.calls
      .map(([u]) => String(u))
      .filter((u) => u.includes("/usage/summary"))
    // The default 30d preset fires a current window (no end_date, "up to now")
    // and a previous window whose end_date is pinned so it does not overlap.
    expect(summaryCalls.some((u) => u.includes("end_date="))).toBe(true)
    expect(summaryCalls.some((u) => !u.includes("end_date="))).toBe(true)
  })

  it("renders period-over-period change as a trend chip, not a glyph", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    // The chip carries the caption the old plain-text hint carried, so the
    // comparison still says what it is being compared against.
    expect(screen.getAllByText(/vs prev/).length).toBeGreaterThan(0)
    // And it announces a direction, which the glyph never did: "▲" is
    // decoration a screen reader skips. TrendChip.test.tsx owns the direction
    // and polarity mapping; this only asserts the tiles go through it.
    // Anchored, and counted. Unanchored, `up` also matches the page's own
    // description and the "No grouping" option, so the assertion would pass with
    // the announcement deleted. The count is the four tiles that have a delta,
    // less cache hit rate, which this fixture gives no input-token composition
    // to divide by.
    expect(
      screen.getAllByText(/^(no change|up|down)(, (better|worse))?$/),
    ).toHaveLength(3)
    // The hand-rolled arrow glyphs are gone from the tiles.
    expect(screen.queryByText(/[▲▼]/)).not.toBeInTheDocument()
  })

  it("reads a chip against the metric's own polarity, not the direction alone", async () => {
    // Spend rose from 827.00 to 1,240.50, a 50% rise. On `down-is-good` that is
    // the regression, so the chip is a danger chip and says so: direction plus
    // judgment, because polarity puts good and bad in hue alone.
    mockApi(
      summary(),
      {},
      summary({
        totals: usageTotals({
          cost: 827,
          request_count: 42_000,
          billed_input_tokens: 4_000_000,
          billed_output_tokens: 2_200_000,
        }),
      }),
    )
    renderPage(<UsagePage />)

    expect(await screen.findByText("up, worse")).toBeInTheDocument()
    expect(screen.getByText("+50.0% vs prev")).toBeInTheDocument()
    // Requests doubled too, but volume carries no polarity, so it announces the
    // direction and nothing more.
    expect(screen.getAllByText("up").length).toBeGreaterThan(0)
  })

  it("renders the trend with recharts and retires the hand-rolled SVG chart", async () => {
    mockApi(summary())
    const { container } = renderPage(<UsagePage />)
    await screen.findByText("gpt-5.6")

    // The trend is now a recharts chart (labeled "<metric> per <bucket>"; a
    // group, not an image, since it owns drag selection), and a reusable
    // sparkline rides the KPI tiles off the same bucketed series.
    expect(
      screen.getByRole("group", { name: "cost per day" }),
    ).toBeInTheDocument()
    expect(screen.getByRole("img", { name: /Spend trend/ })).toBeInTheDocument()
    expect(container.querySelector(".recharts-surface")).not.toBeNull()

    // The retired hand-rolled chart's fingerprints are gone: its "<metric> over
    // time" label and its fixed 720x224 viewBox.
    expect(
      screen.queryByRole("img", { name: /over time/ }),
    ).not.toBeInTheDocument()
    // Presence selector plus a value check: jsdom does not match a camelCase
    // SVG attribute by value, so `svg[viewBox="0 0 720 224"]` would pass here
    // whether or not the retired chart was still rendered.
    expect(
      [...container.querySelectorAll("svg[viewBox]")].map((svg) =>
        svg.getAttribute("viewBox"),
      ),
    ).not.toContain("0 0 720 224")
  })

  it("switches the chart metric via the segmented toggle", async () => {
    const user = userEvent.setup()
    mockApi(summary())
    renderPage(<UsagePage />)

    await screen.findByText("gpt-5.6")
    // Default metric is Cost; the caption shows the peak in dollars.
    expect(screen.getByText(/\$840\.50 peak/)).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Requests" }))
    expect(screen.getByText(/56,000 peak/)).toBeInTheDocument()
  })

  it("shows an onboarding empty state when the gateway has no usage", async () => {
    mockApi(
      summary({
        totals: usageTotals(),
        by_model: [],
        by_user: [],
        series: [],
      }),
    )
    renderPage(<UsagePage />)

    // The default 30d window is the baseline (not a user-applied filter), so an
    // empty gateway reads as onboarding rather than "no rows match".
    expect(await screen.findByText(/No usage yet/)).toBeInTheDocument()
  })

  it("no longer duplicates the Activity page's per-request table", async () => {
    mockApi(summary())
    renderPage(<UsagePage />)
    await screen.findByText("$1,240.50")

    // The per-request table (and its bulk actions) lives on the Activity page;
    // the breakdown rows drill there instead.
    expect(screen.queryByText("Individual requests")).not.toBeInTheDocument()
  })
})
