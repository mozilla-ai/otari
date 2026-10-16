import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query"
import { act, render, screen, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ConnectionStatus } from "@/app/ConnectionStatus"
import { API_ROOT, apiFetch } from "@/shared/api/client"

// Drives one management request so the query cache carries a real success/error,
// exactly what ConnectionStatus watches.
function Probe() {
  useQuery({
    queryKey: ["probe"],
    queryFn: () => apiFetch(`${API_ROOT}/settings`),
    retry: false,
  })
  return null
}

function renderWithProbe(): { client: QueryClient } {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const wrap = (children: ReactNode) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  render(
    wrap(
      <>
        <Probe />
        <ConnectionStatus />
      </>,
    ),
  )
  return { client }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

describe("ConnectionStatus", () => {
  afterEach(() => vi.restoreAllMocks())

  it("alerts when the gateway is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new TypeError("Failed to fetch"),
    )
    renderWithProbe()

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent(/Can’t reach the gateway/)
  })

  it("clears itself once the gateway responds again", async () => {
    let online = false
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      if (!online) throw new TypeError("Failed to fetch")
      return jsonResponse({ ok: true })
    })
    const { client } = renderWithProbe()

    await screen.findByRole("alert")

    online = true
    await client.refetchQueries({ queryKey: ["probe"] })

    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    )
  })

  it("stays quiet for one dropped request that succeeds when asked again", async () => {
    // A laptop waking or a proxy closing an idle connection drops one request;
    // the gateway is fine, so the recheck succeeds and nothing is shown.
    let calls = 0
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      calls += 1
      if (calls === 1) throw new TypeError("Failed to fetch")
      return jsonResponse({ ok: true })
    })
    const { client } = renderWithProbe()

    await waitFor(() =>
      expect(client.getQueryState(["probe"])?.status).toBe("success"),
    )
    expect(calls).toBe(2)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("is not held up by a failure a page left behind once requests succeed", async () => {
    const online = true
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).includes("/left-behind") || !online) {
        throw new TypeError("Failed to fetch")
      }
      return jsonResponse({ ok: true })
    })
    const { client } = renderWithProbe()
    await waitFor(() =>
      expect(client.getQueryState(["probe"])?.status).toBe("success"),
    )

    // A query no page displays any more fails, and keeps failing when rechecked.
    await client
      .fetchQuery({
        queryKey: ["left-behind"],
        queryFn: () => apiFetch(`${API_ROOT}/left-behind`),
      })
      .catch(() => undefined)
    await screen.findByRole("alert")

    // The query stays in error, but the gateway is answering everything else.
    await client.refetchQueries({ queryKey: ["probe"] })

    expect(client.getQueryState(["left-behind"])?.status).toBe("error")
    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    )
  })

  it("keeps the alert through a cache write while the gateway is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new TypeError("Failed to fetch"),
    )
    const { client } = renderWithProbe()
    await screen.findByRole("alert")

    // `setQueryData` reaches no gateway, so it is not an answer from one.
    await act(async () => {
      client.setQueryData(["optimistic"], { ok: true })
    })

    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("stays quiet for an error that is not a lost connection", async () => {
    // A 500 means the backend answered; that is a page-level error, not "offline".
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "boom" }, 500),
    )
    renderWithProbe()

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    )
  })
})
