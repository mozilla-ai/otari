import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, expect, it, vi } from "vitest"

import type { PlatformConnectStart, PlatformConnectStatus } from "@/client"
import { PlatformConnect } from "@/features/providers/PlatformConnect"
import * as apiClient from "@/shared/api/client"

const STARTED: PlatformConnectStart = {
  user_code: "WXYZ-1234",
  verification_uri_complete: "https://otari.test/connect?code=WXYZ-1234",
  expires_in: 600,
}
const PENDING: PlatformConnectStatus = { status: "pending", ...STARTED }

/** The tab the card opens on Connect, as the browser hands it back. */
function fakeTab() {
  const tab = {
    opener: window as Window | null,
    close: vi.fn(),
    location: { replace: vi.fn() },
  }
  vi.spyOn(window, "open").mockReturnValue(tab as unknown as Window)
  return tab
}

/**
 * Answers the connect routes: each status read takes the next of `statuses`
 * (the last one repeating), and every request is recorded as `METHOD path`.
 */
function mockConnect(
  statuses: PlatformConnectStatus[],
  start: () => Promise<PlatformConnectStart> = async () => STARTED,
) {
  const requests: string[] = []
  const queue = [...statuses]
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push(`${method} ${String(input)}`)
    if (method === "POST") return start()
    if (method === "DELETE") return undefined
    return queue.length > 1 ? queue.shift() : queue[0]
  })
  return requests
}

function renderCard(onConnected = vi.fn()) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={queryClient}>
      <PlatformConnect onConnected={onConnected} />
    </QueryClientProvider>,
  )
  return onConnected
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it("says it is paid and that requests leave the gateway, and sends nothing but the status read", async () => {
  const requests = mockConnect([{ status: "idle" }])
  renderCard()

  expect(
    await screen.findByRole("button", { name: "Connect otari.ai" }),
  ).toBeEnabled()
  expect(
    screen.getByText(
      /paid service, and requests sent to it leave this gateway/,
    ),
  ).toBeInTheDocument()
  expect(requests).toEqual(["GET /platform/connect"])
})

it("opens the approval page in a new tab and shows the code, the link and a waiting state in place", async () => {
  const tab = fakeTab()
  const requests = mockConnect([{ status: "idle" }, PENDING])
  renderCard()

  await userEvent.click(
    await screen.findByRole("button", { name: "Connect otari.ai" }),
  )

  expect(await screen.findByText("WXYZ-1234")).toBeInTheDocument()
  expect(requests).toContain("POST /platform/connect")
  expect(window.open).toHaveBeenCalledWith("", "_blank")
  expect(tab.opener).toBeNull()
  expect(tab.location.replace).toHaveBeenCalledWith(
    STARTED.verification_uri_complete,
  )
  const link = screen.getByRole("link", { name: "Open the approval page" })
  expect(link).toHaveAttribute("href", STARTED.verification_uri_complete)
  expect(link).toHaveAttribute("target", "_blank")
  expect(screen.getByRole("status")).toHaveTextContent(
    "This page updates on its own",
  )
  expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled()
})

it("shows the code and its link when the browser blocks the new tab", async () => {
  vi.spyOn(window, "open").mockReturnValue(null)
  mockConnect([{ status: "idle" }, PENDING])
  renderCard()

  await userEvent.click(
    await screen.findByRole("button", { name: "Connect otari.ai" }),
  )

  expect(
    await screen.findByRole("link", { name: "Open the approval page" }),
  ).toBeInTheDocument()
})

it("closes the tab it opened and reports the failure when the gateway cannot start", async () => {
  const tab = fakeTab()
  mockConnect([{ status: "idle" }], async () => {
    throw new apiClient.ApiError(
      503,
      "otari.ai could not be reached. Try again in a moment.",
    )
  })
  renderCard()

  await userEvent.click(
    await screen.findByRole("button", { name: "Connect otari.ai" }),
  )

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "otari.ai could not be reached",
  )
  expect(tab.close).toHaveBeenCalled()
  expect(tab.location.replace).not.toHaveBeenCalled()
})

it("offers no link to an approval page that is not a web address", async () => {
  const tab = fakeTab()
  const unsafe = {
    ...STARTED,
    verification_uri_complete: "javascript:alert(1)",
  }
  mockConnect(
    [{ status: "idle" }, { status: "pending", ...unsafe }],
    async () => unsafe,
  )
  renderCard()

  await userEvent.click(
    await screen.findByRole("button", { name: "Connect otari.ai" }),
  )

  expect(await screen.findByText("WXYZ-1234")).toBeInTheDocument()
  expect(screen.queryByRole("link")).not.toBeInTheDocument()
  expect(tab.location.replace).not.toHaveBeenCalled()
  expect(tab.close).toHaveBeenCalled()
})

it("polls while pending and reports the approval once", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const requests = mockConnect([PENDING, PENDING, { status: "connected" }])
  const onConnected = renderCard()

  expect(await screen.findByText("WXYZ-1234")).toBeInTheDocument()
  expect(onConnected).not.toHaveBeenCalled()

  await vi.advanceTimersByTimeAsync(3_000)
  await vi.advanceTimersByTimeAsync(3_000)

  expect(await screen.findByText(/^Connected\./)).toBeInTheDocument()
  expect(onConnected).toHaveBeenCalledTimes(1)
  const reads = requests.filter((request) => request.startsWith("GET")).length

  // Connected does not change on its own, so the polling stops.
  await vi.advanceTimersByTimeAsync(10_000)
  expect(requests.filter((request) => request.startsWith("GET"))).toHaveLength(
    reads,
  )
  expect(onConnected).toHaveBeenCalledTimes(1)
})

it("does not report a connection that was made before the page opened", async () => {
  const requests = mockConnect([{ status: "connected" }])
  const onConnected = renderCard()

  expect(await screen.findByText(/^Connected\./)).toBeInTheDocument()
  expect(onConnected).not.toHaveBeenCalled()
  expect(
    screen.getByRole("button", { name: "Connect again" }),
  ).toBeInTheDocument()
  // One read, however the provider lists were refreshed around it.
  expect(requests).toEqual(["GET /platform/connect"])
})

it.each([
  ["expired", "The code expired before it was approved."],
  ["denied", "The request was declined on otari.ai."],
] as const)("offers Try again once the attempt is %s", async (status, copy) => {
  fakeTab()
  const requests = mockConnect([{ status }, PENDING])
  renderCard()

  expect(await screen.findByText(copy)).toBeInTheDocument()
  await userEvent.click(screen.getByRole("button", { name: "Try again" }))

  expect(await screen.findByText("WXYZ-1234")).toBeInTheDocument()
  expect(requests).toContain("POST /platform/connect")
  expect(screen.queryByText(copy)).not.toBeInTheDocument()
})

it("cancels a pending attempt and goes back to the offer", async () => {
  const requests = mockConnect([PENDING, { status: "idle" }])
  renderCard()

  await userEvent.click(await screen.findByRole("button", { name: "Cancel" }))

  expect(
    await screen.findByRole("button", { name: "Connect otari.ai" }),
  ).toBeInTheDocument()
  await waitFor(() => expect(requests).toContain("DELETE /platform/connect"))
  expect(screen.queryByText("WXYZ-1234")).not.toBeInTheDocument()
})
