import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { GatewaySettings } from "@/client"
import { PricingWarning } from "@/features/models/PricingWarning"
import { organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const BASE: GatewaySettings = {
  mode: "standalone",
  version: "1.0.0",
  model_discovery: true,
  default_pricing: false,
  require_pricing: true,
  master_key_source: "configured",
  secret_key_configured: true,
  config: [],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockSettings(
  settings: GatewaySettings,
  rejectedInLastHour = 0,
  context = organizationContext(),
) {
  let current = { ...settings }
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      // The shell reads this before it paints, and the usage hooks wait for it:
      // it is what tells them whether this caller reads the deployment-wide
      // routes or the organization-scoped ones (otari#837). Answered first, and
      // on an exact match, so it cannot shadow /v1/organizations/me/usage.
      if (url.endsWith("/v1/organizations/me")) {
        return jsonResponse(context)
      }
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.includes("/v1/settings")) {
        if (method === "PATCH")
          current = { ...current, ...JSON.parse(String(init?.body)) }
        return jsonResponse(current)
      }
      if (url.includes("/v1/usage/count")) {
        return jsonResponse({ total: rejectedInLastHour })
      }
      return jsonResponse({})
    })
}

type FetchMock = ReturnType<typeof mockSettings>

// The `start_date` each /v1/usage/count request asked for, in call order.
function countWindows(fetchMock: FetchMock): string[] {
  return fetchMock.mock.calls
    .map(([u]) => String(u))
    .filter((u) => u.includes("/v1/usage/count"))
    .map((u) => String(new URLSearchParams(u.split("?")[1]).get("start_date")))
}

function renderPage(
  ui: ReactElement,
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) {
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>, {
    wrapper: withRouter(),
  })
  return client
}

// Resolves once the organization context has landed in the cache, which is when
// the operator gate has an answer. `fetch` having been called only says the
// request started, so an absence asserted on that alone would also hold while
// the gate was still undecided.
async function settleContext(client: QueryClient) {
  await waitFor(() =>
    expect(client.getQueryData(["organizations", "context"])).toBeDefined(),
  )
}

describe("PricingWarning", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("alarms and enables default pricing when require_pricing rejects requests", async () => {
    const fetchMock = mockSettings({
      ...BASE,
      require_pricing: true,
      default_pricing: false,
    })
    const user = userEvent.setup()
    renderPage(<PricingWarning />)

    const enable = await screen.findByRole("button", {
      name: "Enable default pricing",
    })
    await user.click(enable)

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes("/v1/settings") && (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body))).toEqual({
      default_pricing: true,
    })
    // Once default pricing is on, the alarm clears.
    await waitFor(() =>
      expect(
        screen.queryByText(/Requests are rejected/),
      ).not.toBeInTheDocument(),
    )
  })

  it("stays quiet when default pricing is already on", async () => {
    mockSettings({ ...BASE, require_pricing: true, default_pricing: true })
    renderPage(<PricingWarning />)

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    expect(screen.queryByText(/Requests are rejected/)).not.toBeInTheDocument()
  })

  it("reports how many requests failed in the last hour and links to them", async () => {
    const fetchMock = mockSettings(
      { ...BASE, require_pricing: true, default_pricing: false },
      12,
    )
    renderPage(<PricingWarning />)

    expect(
      await screen.findByText("12 requests failed in the last hour."),
    ).toBeInTheDocument()
    // The link carries the same scope as the count, so the filtered view it lands
    // on reports the same number the banner just claimed.
    expect(
      screen.getByRole("link", { name: "View failed requests" }),
    ).toHaveAttribute("href", "/activity?status=error&range=1h&source=gateway")

    // Counted from the error rows of the last hour, not from all usage.
    const countUrl = String(
      fetchMock.mock.calls.find(([u]) =>
        String(u).includes("/v1/usage/count"),
      )?.[0],
    )
    expect(new URLSearchParams(countUrl.split("?")[1]).get("status")).toBe(
      "error",
    )
    // Imported usage can carry status=error too; an imported session's failures
    // are not this gateway dropping traffic, so they must not raise the alarm.
    expect(new URLSearchParams(countUrl.split("?")[1]).get("source")).toBe(
      "gateway",
    )
    const since = new Date(countWindows(fetchMock)[0]).getTime()
    expect(Date.now() - since).toBeGreaterThan(0)
    expect(Date.now() - since).toBeLessThanOrEqual(3_600_000 + 5_000)
  })

  it("re-anchors the window as it polls, so a long-open tab still reports the last hour", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const fetchMock = mockSettings(
        { ...BASE, require_pricing: true, default_pricing: false },
        12,
      )
      renderPage(<PricingWarning />)

      await waitFor(() => expect(countWindows(fetchMock)).toHaveLength(1))
      await vi.advanceTimersByTimeAsync(60_000)
      await waitFor(() =>
        expect(countWindows(fetchMock).length).toBeGreaterThan(1),
      )

      // A frozen anchor would ask for the same start_date forever, turning "the
      // last hour" into "everything since this tab was opened".
      const [first, second] = countWindows(fetchMock)
      expect(
        new Date(second).getTime() - new Date(first).getTime(),
      ).toBeGreaterThanOrEqual(59_000)
    } finally {
      vi.useRealTimers()
    }
  })

  it("stays a config note, with no count, while nothing is failing", async () => {
    mockSettings({ ...BASE, require_pricing: true, default_pricing: false }, 0)
    renderPage(<PricingWarning />)

    await screen.findByRole("button", { name: "Enable default pricing" })
    // Settings, the organization context, and the failure count. The context is
    // the third because the count is organization-scoped for a caller who does
    // not operate the deployment and waits to learn which they are (otari#837);
    // in the app it is already cached by the shell. Counted rather than listed
    // because the point is that everything has settled before the absences below
    // are asserted.
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(3))
    expect(
      screen.queryByText(/failed in the last hour/),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("link", { name: "View failed requests" }),
    ).not.toBeInTheDocument()
  })

  it("singularizes a lone failure", async () => {
    mockSettings({ ...BASE, require_pricing: true, default_pricing: false }, 1)
    renderPage(<PricingWarning />)

    expect(
      await screen.findByText("1 request failed in the last hour."),
    ).toBeInTheDocument()
  })

  it("does not count failures while the alarm is quiet", async () => {
    const fetchMock = mockSettings(
      { ...BASE, require_pricing: true, default_pricing: true },
      12,
    )
    renderPage(<PricingWarning />)

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    expect(
      fetchMock.mock.calls.some(([u]) => String(u).includes("/v1/usage/count")),
    ).toBe(false)
  })

  it("asks nothing of the operator-only settings route for a tenant", async () => {
    // The alarm and the button that clears it are both deployment-operator-only,
    // so a caller who does not operate the deployment must not fire the read
    // that feeds a banner they could never be shown (#834).
    const fetchMock = mockSettings(
      { ...BASE, require_pricing: true, default_pricing: false },
      12,
      organizationContext({ deployment_operator: false }),
    )
    const client = renderPage(<PricingWarning />)

    await settleContext(client)
    expect(
      fetchMock.mock.calls.some(([u]) => String(u).includes("/v1/settings")),
    ).toBe(false)
    expect(screen.queryByText(/Requests are rejected/)).not.toBeInTheDocument()
  })

  it("keeps cached settings out of the banner once the caller is not an operator", async () => {
    // A disabled query still serves whatever sits under its key, so a caller
    // demoted mid-session would otherwise keep the banner, and a button that
    // PATCHes a route now refusing them.
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    client.setQueryData(["settings"], {
      ...BASE,
      require_pricing: true,
      default_pricing: false,
    } satisfies GatewaySettings)
    mockSettings(
      { ...BASE, require_pricing: true, default_pricing: false },
      12,
      organizationContext({ deployment_operator: false }),
    )
    renderPage(<PricingWarning />, client)

    await settleContext(client)
    expect(screen.queryByText(/Requests are rejected/)).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Enable default pricing" }),
    ).not.toBeInTheDocument()
  })

  it("can be dismissed", async () => {
    mockSettings({ ...BASE, require_pricing: true, default_pricing: false })
    const user = userEvent.setup()
    renderPage(<PricingWarning />)

    await user.click(await screen.findByRole("button", { name: "Dismiss" }))
    expect(screen.queryByText(/Requests are rejected/)).not.toBeInTheDocument()
  })
})
