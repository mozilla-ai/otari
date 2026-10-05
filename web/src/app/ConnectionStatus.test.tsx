import {
  focusManager,
  onlineManager,
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ConnectionStatus } from "@/app/ConnectionStatus"
import { API_ROOT, apiFetch, DASHBOARD_BUILD_PATH } from "@/shared/api/client"
import { useDashboardBuild } from "@/shared/api/deployment"

const LIVENESS_URL = `${API_ROOT}/health/liveness`

// A page query remains errored when the operator navigates elsewhere. Keep
// that independent of the liveness request so a retry of the original query
// cannot accidentally stand in for gateway recovery.
function Probe() {
  useQuery({
    queryKey: ["probe"],
    queryFn: () => apiFetch("/settings"),
    retry: false,
  })
  return null
}

function BuildPoll() {
  useDashboardBuild()
  return null
}

function createClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false, retryDelay: 0 } },
  })
}

function renderStatus(client = createClient(), withProbe = false) {
  const view = render(
    <QueryClientProvider client={client}>
      {withProbe && <Probe />}
      <ConnectionStatus />
    </QueryClientProvider>,
  )
  return { client, ...view }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

async function waitForLiveness(
  client: QueryClient,
  status: "success" | "error",
) {
  await waitFor(() =>
    expect(client.getQueryState(["gateway-liveness"])?.status).toBe(status),
  )
}

describe("ConnectionStatus", () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    focusManager.setFocused(undefined)
    onlineManager.setOnline(true)
  })

  it("alerts only after repeated failures to reach the gateway", async () => {
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockRejectedValue(new TypeError("Failed to fetch"))
    renderStatus()

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent(/Can’t reach the gateway/)
    expect(fetch).toHaveBeenCalledTimes(3)
    expect(fetch.mock.calls.every(([url]) => url === LIVENESS_URL)).toBe(true)
  })

  it("reports an outage when an intermediary serves HTML with HTTP 200", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(
      async () =>
        new Response("<html><body>Gateway unavailable</body></html>", {
          status: 200,
          headers: { "Content-Type": "text/html" },
        }),
    )
    renderStatus()

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Can’t reach the gateway/,
    )
    expect(fetch).toHaveBeenCalledTimes(3)
  })

  it("reports an outage when the liveness body cannot be decoded", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      const response = new Response("invalid encoding", { status: 200 })
      vi.spyOn(response, "json").mockRejectedValue(
        new TypeError("Failed to decode response body"),
      )
      return response
    })
    renderStatus()

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Can’t reach the gateway/,
    )
    expect(fetch).toHaveBeenCalledTimes(3)
  })

  it.each(["network failure", "HTML fallback", "body decoding failure"])(
    "absorbs a single %s without showing an outage",
    async (failure) => {
      let respond: (response: Response) => void = () => undefined
      const retry = new Promise<Response>((resolve) => {
        respond = resolve
      })
      const fetch = vi.spyOn(globalThis, "fetch")
      if (failure === "network failure") {
        fetch.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      } else if (failure === "HTML fallback") {
        fetch.mockResolvedValueOnce(
          new Response("<html>Gateway unavailable</html>", {
            status: 200,
            headers: { "Content-Type": "text/html" },
          }),
        )
      } else {
        const response = new Response("invalid encoding", { status: 200 })
        vi.spyOn(response, "json").mockRejectedValue(
          new TypeError("Failed to decode response body"),
        )
        fetch.mockResolvedValueOnce(response)
      }
      fetch.mockReturnValueOnce(retry)
      const { client } = renderStatus()

      await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2))
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      await act(async () => respond(jsonResponse("I'm alive!")))
      await waitForLiveness(client, "success")
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    },
  )

  it("does not report an outage while a healthy gateway confirms a page failure", async () => {
    let respond: (response: Response) => void = () => undefined
    const liveness = new Promise<Response>((resolve) => {
      respond = resolve
    })
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      if (url === LIVENESS_URL) return liveness
      throw new TypeError("Failed to fetch")
    })
    const { client } = renderStatus(createClient(), true)
    await waitFor(() =>
      expect(client.getQueryState(["probe"])?.status).toBe("error"),
    )

    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    await act(async () => respond(jsonResponse("I'm alive!")))
    await waitForLiveness(client, "success")
    expect(client.getQueryState(["probe"])?.status).toBe("error")
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("does not report an outage after one dropped build poll", async () => {
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (url) => {
        if (url === DASHBOARD_BUILD_PATH) throw new TypeError("Failed to fetch")
        return jsonResponse("I'm alive!")
      })
    const client = createClient()
    render(
      <QueryClientProvider client={client}>
        <BuildPoll />
        <ConnectionStatus />
      </QueryClientProvider>,
    )
    await waitFor(() =>
      expect(client.getQueryState(["build"])?.status).toBe("error"),
    )
    await waitForLiveness(client, "success")

    expect(
      fetch.mock.calls.filter(([url]) => url === DASHBOARD_BUILD_PATH),
    ).toHaveLength(1)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("does not keep an outage banner from a departed page's cached error", async () => {
    const client = createClient()
    const fetch = vi.spyOn(globalThis, "fetch")
    fetch.mockRejectedValueOnce(new TypeError("Failed to fetch"))
    await expect(
      client.fetchQuery({
        queryKey: ["departed-page"],
        queryFn: () => apiFetch("/settings"),
      }),
    ).rejects.toThrow("Network error")
    fetch.mockImplementation(async () => jsonResponse("I'm alive!"))
    await client.fetchQuery({
      queryKey: ["current-page"],
      queryFn: () => apiFetch("/models"),
    })
    renderStatus(client)
    await waitForLiveness(client, "success")

    expect(
      client
        .getQueryCache()
        .find({ queryKey: ["departed-page"] })
        ?.isActive(),
    ).toBe(false)
    expect(client.getQueryState(["departed-page"])?.status).toBe("error")
    expect(client.getQueryState(["current-page"])?.status).toBe("success")
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("polls for recovery even when the failed page query stays in the cache", async () => {
    vi.useFakeTimers()
    let online = false
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      if (!online || url !== LIVENESS_URL)
        throw new TypeError("Failed to fetch")
      return jsonResponse("I'm alive!")
    })
    const { client } = renderStatus(createClient(), true)
    await advance(20)
    expect(screen.getByRole("alert")).toHaveTextContent(
      /Can’t reach the gateway/,
    )

    online = true
    await advance(15_000)

    expect(client.getQueryState(["gateway-liveness"])?.status).toBe("success")
    expect(client.getQueryState(["probe"])?.status).toBe("error")
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("rechecks immediately when the operator returns to the tab", async () => {
    focusManager.setFocused(false)
    let online = false
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      if (!online) throw new TypeError("Failed to fetch")
      return jsonResponse("I'm alive!")
    })
    renderStatus()
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))
    // Retries pause in a background tab. Bring it forward to finish confirming
    // the outage, then test the next focus transition as the recovery trigger.
    await act(async () => focusManager.setFocused(true))
    await screen.findByRole("alert")
    await act(async () => focusManager.setFocused(false))

    online = true
    await act(async () => focusManager.setFocused(true))
    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    )
  })

  it("confirms an outage even when the browser reports being offline", async () => {
    onlineManager.setOnline(false)
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new TypeError("Failed to fetch"),
    )
    renderStatus()
    await screen.findByRole("alert")
  })

  it("rechecks immediately when the browser reconnects", async () => {
    onlineManager.setOnline(false)
    let online = false
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      if (!online) throw new TypeError("Failed to fetch")
      return jsonResponse("I'm alive!")
    })
    renderStatus()
    await screen.findByRole("alert")

    online = true
    await act(async () => onlineManager.setOnline(true))
    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    )
  })

  it.each([401, 403, 500])(
    "stays quiet when the gateway answers HTTP %s",
    async (status) => {
      const fetch = vi
        .spyOn(globalThis, "fetch")
        .mockImplementation(async () =>
          jsonResponse({ detail: "refused" }, status),
        )
      const { client } = renderStatus()
      await waitForLiveness(client, "error")

      expect(fetch).toHaveBeenCalledTimes(1)
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    },
  )

  it("stops polling when the connection banner's owner unmounts", async () => {
    vi.useFakeTimers()
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => jsonResponse("I'm alive!"))
    const { unmount } = renderStatus()
    await advance(20)
    expect(fetch).toHaveBeenCalledTimes(1)

    unmount()
    await advance(30_000)
    expect(fetch).toHaveBeenCalledTimes(1)
  })
})
