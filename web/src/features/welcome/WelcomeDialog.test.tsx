import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  PlatformConnectStart,
  PlatformConnectStatus,
} from "@/client"
import {
  PLATFORM_PLANS_URL,
  WELCOME_ANSWERED_KEY,
  WelcomeDialog,
} from "@/features/welcome/WelcomeDialog"
import * as apiClient from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { flushRouter, renderWithRouter } from "@/tests/router"

const STARTED: PlatformConnectStart = {
  user_code: "WXYZ-1234",
  verification_uri_complete:
    "https://otari.test/dashboard/?region=us#/connect?code=WXYZ-1234",
  expires_in: 600,
}

/**
 * Answers the reads the dialog makes, recording each as `METHOD path`. Each
 * connect status read takes the next of `statuses`, the last one repeating.
 */
function mockApi({
  context = organizationContext(),
  hostedNames = [],
  statuses = [{ status: "idle" }],
}: {
  context?: OrganizationContext
  hostedNames?: string[]
  statuses?: PlatformConnectStatus[]
} = {}) {
  const requests: string[] = []
  const queue = [...statuses]
  vi.spyOn(apiClient, "apiFetch").mockImplementation(async (input, init) => {
    const path = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push(`${method} ${path}`)
    if (path === "/organizations/me") return context
    if (path.startsWith("/hosted-providers")) {
      return { data: hostedNames.map((name) => ({ name })) }
    }
    if (path === "/platform/connect") {
      if (method === "POST") return STARTED
      return queue.length > 1 ? queue.shift() : queue[0]
    }
    return {}
  })
  return requests
}

async function renderDialog(platformConnect = true) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  await renderWithRouter(
    <QueryClientProvider client={queryClient}>
      <DeploymentProvider
        value={bootstrap({ platform_connect: platformConnect })}
      >
        <WelcomeDialog />
      </DeploymentProvider>
    </QueryClientProvider>,
    { routes: [{ path: "/", element: <p>OVERVIEW</p> }], url: "/providers" },
  )
}

const TITLE = "Welcome to Otari"

describe("WelcomeDialog", () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it("greets an operator, links the plans, and sends nothing to otari.ai", async () => {
    const requests = mockApi()
    await renderDialog()

    const dialog = await screen.findByRole("dialog", { name: TITLE })
    expect(dialog).toHaveTextContent("Open-weight models")
    expect(dialog).toHaveTextContent("glm-5.3")
    expect(screen.getByRole("link", { name: "See plans" })).toHaveAttribute(
      "href",
      PLATFORM_PLANS_URL,
    )
    expect(
      await screen.findByRole("button", { name: "Connect otari.ai" }),
    ).toBeEnabled()
    expect(requests.filter((r) => !r.startsWith("GET "))).toEqual([])
  })

  it.each([
    ["the deployment does not offer otari.ai", { platformConnect: false }],
    [
      "the caller is not the operator",
      { context: organizationContext({ deployment_operator: false }) },
    ],
    ["the gateway is already connected", { hostedNames: ["otari-ai"] }],
    [
      "no key can be stored",
      {
        context: organizationContext({
          provider_key_encryption_available: false,
        }),
      },
    ],
  ])("stays away when %s", async (_case, options) => {
    const { platformConnect = true, ...api } = options as {
      platformConnect?: boolean
      context?: OrganizationContext
      hostedNames?: string[]
    }
    const requests = mockApi(api)
    await renderDialog(platformConnect)

    if (platformConnect) {
      await waitFor(() => expect(requests).toContain("GET /organizations/me"))
    } else {
      expect(requests).toEqual([])
    }
    await flushRouter()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("asks nothing once answered in this browser", async () => {
    localStorage.setItem(WELCOME_ANSWERED_KEY, "2026-11-13T00:00:00Z")
    const requests = mockApi()
    await renderDialog()

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(requests).toEqual([])
  })

  it.each(["Skip, I’ll use my own keys", "Close"])(
    "records %s as an answer and closes",
    async (label) => {
      mockApi()
      const user = userEvent.setup()
      await renderDialog()

      await screen.findByRole("dialog", { name: TITLE })
      await user.click(screen.getByRole("button", { name: label }))

      await waitFor(() =>
        expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
      )
      expect(localStorage.getItem(WELCOME_ANSWERED_KEY)).not.toBeNull()
    },
  )

  it("shows the code to approve while otari.ai waits, and Cancel ends it", async () => {
    const requests = mockApi({
      statuses: [{ status: "idle" }, { status: "pending", ...STARTED }],
    })
    vi.spyOn(window, "open").mockReturnValue(null)
    const user = userEvent.setup()
    await renderDialog()

    await user.click(
      await screen.findByRole("button", { name: "Connect otari.ai" }),
    )

    expect(await screen.findByText("WXYZ-1234")).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Open the approval page" }),
    ).toHaveAttribute("href", STARTED.verification_uri_complete)
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(requests).toContain("DELETE /platform/connect"))
  })

  it("closes and opens the Overview once the connection is approved", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    mockApi({
      statuses: [
        { status: "idle" },
        { status: "pending", ...STARTED },
        { status: "connected" },
      ],
    })
    const tab = {
      opener: window as Window | null,
      close: vi.fn(),
      location: { replace: vi.fn() },
    }
    vi.spyOn(window, "open").mockReturnValue(tab as unknown as Window)
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    await renderDialog()

    await user.click(
      await screen.findByRole("button", { name: "Connect otari.ai" }),
    )
    expect(await screen.findByText("WXYZ-1234")).toBeInTheDocument()
    await vi.advanceTimersByTimeAsync(3_000)
    await vi.advanceTimersByTimeAsync(3_000)

    expect(await screen.findByText("OVERVIEW")).toBeInTheDocument()
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
    expect(localStorage.getItem(WELCOME_ANSWERED_KEY)).not.toBeNull()
  })

  it("closes the blank approval tab when dismissed before the gateway answers", async () => {
    mockApi()
    vi.spyOn(apiClient, "apiFetch").mockImplementation(async (input, init) => {
      const path = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (path === "/organizations/me") return organizationContext()
      if (path.startsWith("/hosted-providers")) return { data: [] }
      if (path === "/platform/connect" && method === "POST") {
        return new Promise(() => {})
      }
      return { status: "idle" }
    })
    const tab = {
      opener: window as Window | null,
      close: vi.fn(),
      location: { replace: vi.fn() },
    }
    vi.spyOn(window, "open").mockReturnValue(tab as unknown as Window)
    const user = userEvent.setup()
    await renderDialog()

    await user.click(
      await screen.findByRole("button", { name: "Connect otari.ai" }),
    )
    await user.click(
      screen.getByRole("button", { name: "Skip, I’ll use my own keys" }),
    )

    await waitFor(() => expect(tab.close).toHaveBeenCalled())
  })
})
