import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { HealthCheckSettings, HealthCheckStatus } from "@/client"
import { HealthChecksPage } from "@/features/health/HealthChecksPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

interface Request {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function settings(
  overrides: Partial<HealthCheckSettings> = {},
): HealthCheckSettings {
  return {
    schedule_enabled: true,
    interval_minutes: 60,
    prompt: "Reply with the single word: pong",
    max_tokens: 16,
    reasoning_max_tokens: 256,
    reasoning_every_runs: 6,
    next_run_at: "2026-10-16T10:00:00Z",
    run_requested: false,
    alert_emails: [],
    ...overrides,
  }
}

const RUN = {
  id: "run-1",
  started_at: "2026-10-16T09:00:00Z",
  finished_at: "2026-10-16T09:01:00Z",
  included_reasoning: false,
  total: 3,
  passed: 1,
  failed: 2,
}

function status(overrides: Partial<HealthCheckStatus> = {}): HealthCheckStatus {
  return {
    settings: settings(),
    running: null,
    last_run: RUN,
    providers: [
      {
        provider: "bedrock",
        status: "down",
        answered: 0,
        total: 2,
        top_error: "[bedrock] model <model> is not available",
        top_error_count: 2,
        history: [],
      },
      {
        provider: "deepinfra",
        status: "ok",
        answered: 1,
        total: 1,
        top_error: null,
        top_error_count: 0,
        history: [],
      },
    ],
    results: [
      {
        selector: "bedrock:claude",
        provider: "bedrock",
        status: "fail",
        status_code: 404,
        error: "[bedrock] model claude is not available",
        latency_ms: 120,
        checked_at: "2026-10-16T09:01:00Z",
        failing_since: "2026-10-16T06:00:00Z",
        failed_runs: 4,
      },
      {
        selector: "deepinfra:llama",
        provider: "deepinfra",
        status: "pass",
        status_code: null,
        error: null,
        latency_ms: 300,
        checked_at: "2026-10-16T09:01:00Z",
        failing_since: null,
        failed_runs: 0,
      },
    ],
    ...overrides,
  }
}

function mockApi(
  answer: HealthCheckStatus | { $status: number },
  { afterDisable }: { afterDisable?: HealthCheckStatus } = {},
) {
  const requests: Request[] = []
  let disabled = false
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    requests.push({ url, method, body })
    if (url.endsWith(`${API_ROOT}/health-checks/disable-failing`)) {
      disabled = true
      return jsonResponse({ disabled: 1 })
    }
    if (url.endsWith(`${API_ROOT}/health-checks/run`)) {
      return jsonResponse(settings({ run_requested: true }), 202)
    }
    if (url.endsWith(`${API_ROOT}/health-checks/settings`)) {
      return jsonResponse(settings(body))
    }
    if (url.endsWith(`${API_ROOT}/health-checks`)) {
      return "$status" in answer
        ? jsonResponse(
            { detail: "This endpoint requires deployment operator access." },
            answer.$status,
          )
        : jsonResponse(disabled && afterDisable ? afterDisable : answer)
    }
    return jsonResponse({})
  })
  return requests
}

function renderPage({ mailReady = true } = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap({ mail_ready: mailReady })}>
      <QueryClientProvider client={client}>
        <HealthChecksPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/health-checks" },
  )
}

afterEach(() => vi.restoreAllMocks())

describe("HealthChecksPage", () => {
  it("names the provider that is failing, worst first", async () => {
    mockApi(status())
    await renderPage()

    const providers = await screen.findByRole("grid", { name: "Providers" })
    const rows = within(providers).getAllByRole("row").slice(1)
    expect(within(rows[0]).getByText("bedrock")).toBeInTheDocument()
    expect(within(rows[0]).getByText("Down")).toBeInTheDocument()
    expect(within(rows[0]).getByText("0/2")).toBeInTheDocument()
    expect(
      within(rows[0]).getByText(/2× \[bedrock\] model <model>/),
    ).toBeInTheDocument()
    expect(within(rows[1]).getByText("OK")).toBeInTheDocument()
  })

  it("lists only the failing models until asked for all", async () => {
    mockApi(status())
    await renderPage()

    const models = await screen.findByRole("grid", { name: "Models" })
    expect(within(models).getByText("bedrock:claude")).toBeInTheDocument()
    expect(within(models).getByText("HTTP 404")).toBeInTheDocument()
    expect(within(models).getByText(/· 4 runs$/)).toBeInTheDocument()
    expect(within(models).queryByText("deepinfra:llama")).toBeNull()

    await userEvent.click(screen.getByRole("button", { name: "Show all" }))

    expect(
      within(screen.getByRole("grid", { name: "Models" })).getByText(
        "deepinfra:llama",
      ),
    ).toBeInTheDocument()
  })

  it("pages the failing models", async () => {
    const failing = Array.from({ length: 30 }, (_, i) => ({
      ...status().results[0],
      selector: `bedrock:model-${String(i).padStart(2, "0")}`,
    }))
    mockApi(status({ results: failing }))
    await renderPage()

    const models = await screen.findByRole("grid", { name: "Models" })
    expect(within(models).getAllByRole("row").slice(1)).toHaveLength(25)
    expect(within(models).queryByText("bedrock:model-25")).toBeNull()

    await userEvent.click(screen.getByRole("button", { name: "Next page" }))

    const second = screen.getByRole("grid", { name: "Models" })
    expect(within(second).getAllByRole("row").slice(1)).toHaveLength(5)
    expect(within(second).getByText("bedrock:model-29")).toBeInTheDocument()
  })

  it("keeps the pager away while every failing model fits one page", async () => {
    mockApi(status())
    await renderPage()

    await screen.findByRole("grid", { name: "Models" })
    expect(screen.queryByRole("button", { name: "Next page" })).toBeNull()
  })

  it("switches off every failing model after a confirmation", async () => {
    const requests = mockApi(status(), {
      afterDisable: status({
        results: status().results.filter((r) => r.status !== "fail"),
      }),
    })
    await renderPage()

    await userEvent.click(
      await screen.findByRole("button", { name: "Disable all" }),
    )
    expect(
      requests.some((r) => r.url.endsWith("/health-checks/disable-failing")),
    ).toBe(false)
    await userEvent.click(
      screen.getByRole("button", { name: "Disable 1 model" }),
    )

    await waitFor(() =>
      expect(
        requests.filter(
          (r) =>
            r.method === "POST" &&
            r.url.endsWith("/health-checks/disable-failing"),
        ),
      ).toHaveLength(1),
    )
    expect(await screen.findByText(/1 model switched off/)).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByText("bedrock:claude")).toBeNull())
    expect(screen.queryByRole("button", { name: "Disable all" })).toBeNull()
  })

  it("starts a run on request and says it is starting", async () => {
    const requests = mockApi(status())
    await renderPage()

    await userEvent.click(
      await screen.findByRole("button", { name: "Run now" }),
    )

    await waitFor(() =>
      expect(
        requests.some(
          (r) => r.method === "POST" && r.url.endsWith("/health-checks/run"),
        ),
      ).toBe(true),
    )
    expect(
      await screen.findByRole("button", { name: "Running…" }),
    ).toBeDisabled()
    expect(screen.getByText(/Starting a run/)).toBeInTheDocument()
  })

  it("saves the schedule", async () => {
    const requests = mockApi(status())
    await renderPage()

    const interval = await screen.findByLabelText("Minutes between runs")
    await userEvent.clear(interval)
    await userEvent.type(interval, "30")
    await userEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      const patch = requests.find((r) => r.method === "PATCH")
      expect(patch?.body).toMatchObject({ interval_minutes: 30 })
    })
    expect(await screen.findByText("Saved")).toBeInTheDocument()
  })

  it.each([
    ["", "Enter a whole number."],
    ["30m", "Enter a whole number."],
    ["5", "Between 15 and 1440."],
  ])("refuses to save a schedule interval of %j", async (value, problem) => {
    const requests = mockApi(status())
    await renderPage()

    const interval = await screen.findByLabelText("Minutes between runs")
    await userEvent.clear(interval)
    if (value) await userEvent.type(interval, value)

    expect(await screen.findByText(problem)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(requests.some((r) => r.method === "PATCH")).toBe(false)
  })

  it("stops saying saved once the schedule is edited again", async () => {
    mockApi(status())
    await renderPage()

    const interval = await screen.findByLabelText("Minutes between runs")
    await userEvent.clear(interval)
    await userEvent.type(interval, "30")
    await userEvent.click(screen.getByRole("button", { name: "Save" }))
    expect(await screen.findByText("Saved")).toBeInTheDocument()

    await userEvent.type(interval, "0")

    expect(screen.queryByText("Saved")).toBeNull()
  })

  it("keeps Save disabled while a save is in flight, even after an edit", async () => {
    const requests = mockApi(status())
    const fetchSpy = vi.mocked(globalThis.fetch)
    const answer = fetchSpy.getMockImplementation()
    let release: () => void = () => {}
    fetchSpy.mockImplementation(async (input, init) => {
      if ((init?.method ?? "GET").toUpperCase() === "PATCH") {
        await new Promise<void>((resolve) => {
          release = resolve
        })
      }
      return answer?.(input, init) ?? new Response("{}")
    })
    await renderPage()

    const interval = await screen.findByLabelText("Minutes between runs")
    await userEvent.clear(interval)
    await userEvent.type(interval, "30")
    await userEvent.click(screen.getByRole("button", { name: "Save" }))
    const saving = await screen.findByRole("button", { name: "Saving…" })

    await userEvent.type(interval, "0")

    expect(saving).toBeDisabled()
    release()
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Save" })).toBeEnabled(),
    )
    // The form now holds 300, which was never sent, so it does not read as saved.
    expect(screen.queryByText("Saved")).toBeNull()
    expect(requests.filter((r) => r.method === "PATCH")).toHaveLength(1)
  })

  it("says so before the first run", async () => {
    mockApi(status({ last_run: null, providers: [], results: [] }))
    await renderPage()

    expect(
      await screen.findByText(/No run has finished yet/),
    ).toBeInTheDocument()
    expect(
      screen.getByText("No results yet. Run the checks to see each provider."),
    ).toBeInTheDocument()
  })

  it("shows the refusal a non-operator gets", async () => {
    mockApi({ $status: 403 })
    await renderPage()

    expect(
      await screen.findByText(
        "This endpoint requires deployment operator access.",
      ),
    ).toBeInTheDocument()
    expect(screen.queryByRole("grid")).toBeNull()
  })

  describe("email alerts", () => {
    it("saves the list, lowercased and once each", async () => {
      const requests = mockApi(status())
      await renderPage()

      await userEvent.type(
        await screen.findByLabelText("Send alerts to"),
        "Ops@Example.com, oncall@example.com ops@example.com",
      )
      await userEvent.click(
        screen.getByRole("button", { name: "Save addresses" }),
      )

      await waitFor(() => {
        const patch = requests.find((r) => r.method === "PATCH")
        expect(patch?.body).toEqual({
          alert_emails: ["ops@example.com", "oncall@example.com"],
        })
      })
    })

    it("names an address it could not send to, before saving", async () => {
      const requests = mockApi(status())
      await renderPage()

      await userEvent.type(
        await screen.findByLabelText("Send alerts to"),
        "ops@example.com, oncall",
      )

      expect(
        screen.getByText("Not an email address: oncall"),
      ).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: "Save addresses" }),
      ).toBeDisabled()
      expect(requests.some((r) => r.method === "PATCH")).toBe(false)
    })

    it("says alerts go nowhere while the deployment cannot send mail", async () => {
      mockApi(
        status({ settings: settings({ alert_emails: ["ops@example.com"] }) }),
      )
      await renderPage({ mailReady: false })

      expect(
        await screen.findByText(/This deployment can't send email yet/),
      ).toBeInTheDocument()
      expect(screen.getByRole("link", { name: "Settings" })).toBeInTheDocument()
    })
  })
})
