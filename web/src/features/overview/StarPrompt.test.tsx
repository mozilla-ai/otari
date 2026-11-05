import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap, UsageSummary } from "@/client"
import { StarPrompt } from "@/features/overview/StarPrompt"
import { STAR_PROMPT_STORE_KEY } from "@/features/overview/starPrompt"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import {
  bootstrap,
  organizationContext,
  seriesPoint,
  usageTotals,
} from "@/tests/fixtures"
import { recordEvent } from "@/tests/telemetry"

vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  })
}

/** The successes the invitation reads, one count per day bucket. */
function successes(perDay: number[]): UsageSummary {
  return {
    start_date: "2026-10-04T00:00:00Z",
    end_date: "2026-11-02T00:00:00Z",
    bucket: "day",
    totals: usageTotals({
      request_count: perDay.reduce((sum, count) => sum + count, 0),
    }),
    by_model: [],
    by_user: [],
    by_api_key: [],
    by_source: [],
    by_source_label: [],
    by_endpoint: [],
    by_provider: [],
    by_tool: [],
    by_tag: [],
    errors_by_status_code: [],
    series: perDay.map((requests, day) =>
      seriesPoint({
        bucket_start: `2026-10-${String(day + 4).padStart(2, "0")}T00:00:00Z`,
        requests,
      }),
    ),
  }
}

const EARNED = successes([20, 0, 5])

function mockApi(answer: UsageSummary = EARNED) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(organizationContext())
    }
    if (url.includes("/usage/summary")) {
      return jsonResponse(answer)
    }
    return jsonResponse([])
  })
}

/** The usage reads the invitation made, which is none when a gate said no. */
const summaryReads = (fetchMock: ReturnType<typeof mockApi>) =>
  fetchMock.mock.calls
    .map(([input]) => String(input))
    .filter((url) => url.includes("/usage/summary"))

function renderPrompt({
  canAsk = true,
  isQuiet = true,
  deployment = {},
}: {
  canAsk?: boolean
  isQuiet?: boolean
  deployment?: Partial<DeploymentBootstrap>
} = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap(deployment)}>
        <main id="main-content" tabIndex={-1}>
          <StarPrompt canAsk={canAsk} isQuiet={isQuiet} />
        </main>
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

const LINK = { name: "Star Otari on GitHub" }
const CLOSE = { name: "Dismiss the invitation to star Otari" }

describe("the invitation to star Otari", () => {
  beforeEach(() => {
    window.localStorage.removeItem(STAR_PROMPT_STORE_KEY)
    recordEvent.mockClear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("asks once the gateway has earned it, in one line with a way out", async () => {
    mockApi()
    renderPrompt()

    const link = await screen.findByRole("link", LINK)
    expect(link).toHaveAttribute("href", "https://github.com/mozilla-ai/otari")
    expect(link).toHaveAttribute("target", "_blank")
    expect(
      screen.getByText(
        /Otari is open source\. A star helps other developers find it\./,
      ),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", CLOSE)).toBeInTheDocument()
    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.STAR_PROMPT_SHOWN)
    expect(recordEvent).toHaveBeenCalledTimes(1)
  })

  it("reads successes only, by day, from the start of a thirty-day window", async () => {
    const fetchMock = mockApi()
    renderPrompt()

    await screen.findByRole("link", LINK)
    const [read] = summaryReads(fetchMock)
    const params = new URL(read, "http://localhost").searchParams
    expect(params.get("status")).toBe("success")
    expect(params.get("bucket")).toBe("day")
    const since = new Date(params.get("start_date") ?? "")
    // Local midnight 29 days back, so today counts as the thirtieth. Bounded
    // loosely enough that a daylight-saving change inside the window, which
    // moves the span by an hour, does not fail it.
    const days = (Date.now() - since.getTime()) / 86_400_000
    expect(days).toBeGreaterThan(28.9)
    expect(days).toBeLessThan(30.1)
  })

  it("says nothing until the usage has earned it", async () => {
    const fetchMock = mockApi(successes([400]))
    renderPrompt()

    await waitFor(() => expect(summaryReads(fetchMock)).toHaveLength(1))
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(screen.queryByRole("link", LINK)).not.toBeInTheDocument()
    expect(recordEvent).not.toHaveBeenCalled()
  })

  it("stays away while the page has something else to say", async () => {
    const fetchMock = mockApi()
    renderPrompt({ isQuiet: false })

    await waitFor(() => expect(summaryReads(fetchMock)).toHaveLength(1))
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(screen.queryByRole("link", LINK)).not.toBeInTheDocument()
    expect(recordEvent).not.toHaveBeenCalled()
  })

  it("asks no one, and reads nothing, where the operator withdrew it", async () => {
    const fetchMock = mockApi()
    renderPrompt({ deployment: { star_prompt_enabled: false } })

    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(screen.queryByRole("link", LINK)).not.toBeInTheDocument()
    expect(summaryReads(fetchMock)).toEqual([])
  })

  it("asks no one, and reads nothing, for a caller who did not choose the gateway", async () => {
    const fetchMock = mockApi()
    renderPrompt({ canAsk: false })

    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(screen.queryByRole("link", LINK)).not.toBeInTheDocument()
    expect(summaryReads(fetchMock)).toEqual([])
  })

  it("never asks a browser that already answered", async () => {
    window.localStorage.setItem(STAR_PROMPT_STORE_KEY, "starred")
    const fetchMock = mockApi()
    renderPrompt()

    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(screen.queryByRole("link", LINK)).not.toBeInTheDocument()
    expect(summaryReads(fetchMock)).toEqual([])
  })

  it("is put away for good by its close control, leaving focus on the page", async () => {
    mockApi()
    renderPrompt()

    await userEvent.click(await screen.findByRole("button", CLOSE))

    expect(screen.queryByRole("link", LINK)).not.toBeInTheDocument()
    expect(window.localStorage.getItem(STAR_PROMPT_STORE_KEY)).toBe("dismissed")
    expect(recordEvent).toHaveBeenLastCalledWith(
      TELEMETRY_EVENTS.STAR_PROMPT_DISMISSED,
    )
    expect(document.activeElement).toBe(document.getElementById("main-content"))
  })

  it("is retired by following its link, which counts as the answer", async () => {
    mockApi()
    renderPrompt()

    const link = await screen.findByRole("link", LINK)
    // jsdom does not navigate, so the click is the whole of what is observable.
    await userEvent.click(link)

    expect(screen.queryByRole("link", LINK)).not.toBeInTheDocument()
    expect(window.localStorage.getItem(STAR_PROMPT_STORE_KEY)).toBe("starred")
    expect(recordEvent).toHaveBeenLastCalledWith(
      TELEMETRY_EVENTS.STAR_PROMPT_CLICKED,
    )
  })
})
