import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Outlet } from "@tanstack/react-router"
import { act, renderHook, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationContext,
  PlatformConnectStart,
  PlatformConnectStatus,
} from "@/client"
import { SetupGuide } from "@/features/onboarding/SetupGuide"
import { WelcomeDialog } from "@/features/welcome/WelcomeDialog"
import {
  OWN_KEYS,
  WELCOME_ANSWERED_KEY,
} from "@/features/welcome/welcomeAnswer"
import * as apiClient from "@/shared/api/client"
import {
  useHostedProviders,
  usePlatformConnectStatus,
} from "@/shared/api/hostedProviders"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  workspaceActivation,
} from "@/tests/fixtures"
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
  configuredNames = [],
  statuses = [{ status: "idle" }],
}: {
  context?: OrganizationContext
  hostedNames?: string[]
  configuredNames?: string[]
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
    if (path === "/providers") {
      return { providers: configuredNames.map((instance) => ({ instance })) }
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
  return queryClient
}

const TITLE = "Welcome to Otari"

describe("WelcomeDialog", () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it("greets an operator without a plans link and sends nothing to otari.ai", async () => {
    const requests = mockApi()
    await renderDialog()

    const dialog = await screen.findByRole("dialog", { name: TITLE })
    expect(dialog).toHaveTextContent("Open-weight models")
    expect(dialog).toHaveTextContent("glm-5.3")
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull()
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
    ["a hosted provider already serves", { hostedNames: ["openai"] }],
    [
      "config.yml already configures a provider",
      { configuredNames: ["openai"] },
    ],
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
      configuredNames?: string[]
    }
    const requests = mockApi(api)
    await renderDialog(platformConnect)

    if (platformConnect) {
      await waitFor(() => expect(requests).toContain("GET /organizations/me"))
    } else {
      expect(requests).toEqual([])
    }
    // Long enough for every read the decision waits on to have answered, so a
    // dialog that would open is on screen before the assertion below.
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)))
    await flushRouter()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("waits for config.yml's providers when they answer after the hosted list", async () => {
    const requests = mockApi({ configuredNames: ["openai"] })
    const served = vi.mocked(apiClient.apiFetch).getMockImplementation()
    let answerProviders = () => {}
    vi.mocked(apiClient.apiFetch).mockImplementation((input, init) =>
      String(input) === "/providers"
        ? new Promise((resolve) => {
            answerProviders = () =>
              resolve({ providers: [{ instance: "openai" }] })
          })
        : (served?.(input, init) ?? Promise.resolve({})),
    )
    await renderDialog()

    await waitFor(() =>
      expect(requests.some((r) => r.startsWith("GET /hosted-providers"))).toBe(
        true,
      ),
    )
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    await act(async () => answerProviders())
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it.each(["2026-11-13T00:00:00Z", OWN_KEYS, "e2e"])(
    "asks nothing once answered %s in this browser",
    async (answer) => {
      localStorage.setItem(WELCOME_ANSWERED_KEY, answer)
      const requests = mockApi()
      await renderDialog()

      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      expect(requests).toEqual([])
    },
  )

  it.each([
    ["Skip, I’ll use my own keys", OWN_KEYS],
    ["Close", expect.stringMatching(/^\d{4}-\d{2}-\d{2}T/)],
  ])("records %s as its own answer and closes", async (label, recorded) => {
    mockApi()
    const user = userEvent.setup()
    await renderDialog()

    await screen.findByRole("dialog", { name: TITLE })
    await user.click(screen.getByRole("button", { name: label }))

    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
    expect(localStorage.getItem(WELCOME_ANSWERED_KEY)).toEqual(recorded)
  })

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

  it.each(["Skip, I’ll use my own keys", "Close"])(
    "ends a code still waiting for approval on %s",
    async (label) => {
      const requests = mockApi({
        statuses: [{ status: "idle" }, { status: "pending", ...STARTED }],
      })
      vi.spyOn(window, "open").mockReturnValue(null)
      const user = userEvent.setup()
      const queryClient = await renderDialog()

      await user.click(
        await screen.findByRole("button", { name: "Connect otari.ai" }),
      )
      expect(await screen.findByText("WXYZ-1234")).toBeInTheDocument()
      await user.click(screen.getByRole("button", { name: label }))

      await waitFor(() =>
        expect(requests).toContain("DELETE /platform/connect"),
      )
      // The status the providers page reads to decide whether to show the
      // code, taken from the cache without another read.
      const { result } = renderHook(() => usePlatformConnectStatus(false), {
        wrapper: ({ children }) => (
          <QueryClientProvider client={queryClient}>
            {children}
          </QueryClientProvider>
        ),
      })
      await waitFor(() => expect(result.current.data?.status).toBe("idle"))
    },
  )

  it("closes and opens the Overview once the connection is approved", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const requests = mockApi({
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
    expect(localStorage.getItem(WELCOME_ANSWERED_KEY)).not.toBe(OWN_KEYS)
    expect(localStorage.getItem(WELCOME_ANSWERED_KEY)).not.toBeNull()
    expect(requests).not.toContain("DELETE /platform/connect")
  })

  it("lands on the Overview, where the setup guide opens over the provider it connected", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const workspace = "44444444-4444-4444-4444-444444444444"
    const statuses: PlatformConnectStatus[] = [
      { status: "idle" },
      { status: "pending", ...STARTED },
      { status: "connected" },
    ]
    let connected = false
    vi.spyOn(apiClient, "apiFetch").mockImplementation(async (input, init) => {
      const path = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (path === "/organizations/me") {
        return organizationContext({
          workspace_memberships: [
            {
              workspace_id: workspace,
              name: "Default Workspace",
              role: "owner",
            },
          ],
        })
      }
      if (path.startsWith("/hosted-providers")) {
        return { data: connected ? [{ name: "otari-ai" }] : [] }
      }
      if (path === "/platform/connect") {
        if (method === "POST") return STARTED
        const next = statuses.length > 1 ? statuses.shift() : statuses[0]
        connected = next?.status === "connected"
        return next
      }
      if (path.endsWith("/activation")) return workspaceActivation()
      if (path.startsWith("/models")) return { object: "list", data: [] }
      return {}
    })
    const tab = {
      opener: window as Window | null,
      close: vi.fn(),
      location: { replace: vi.fn() },
    }
    vi.spyOn(window, "open").mockReturnValue(tab as unknown as Window)

    // The operator Overview's guide: held back until a provider can serve.
    function Overview() {
      const hosted = useHostedProviders()
      return (
        <SetupGuide canServeRequests={(hosted.data?.data ?? []).length > 0} />
      )
    }
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    // The shell holds the providers and the welcome; the routes render below.
    await renderWithRouter(<p>PROVIDERS</p>, {
      url: "/providers",
      routes: [{ path: "/", element: <Overview /> }],
      shell: (
        <QueryClientProvider client={queryClient}>
          <DeploymentProvider value={bootstrap({ platform_connect: true })}>
            <SelectedWorkspaceProvider>
              <WelcomeDialog />
              <Outlet />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </QueryClientProvider>
      ),
    })
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })

    await user.click(
      await screen.findByRole("button", { name: "Connect otari.ai" }),
    )
    expect(await screen.findByText("WXYZ-1234")).toBeInTheDocument()
    await vi.advanceTimersByTimeAsync(3_000)
    await vi.advanceTimersByTimeAsync(3_000)

    expect(
      await screen.findByRole("heading", { name: "Send your first request" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("dialog", { name: TITLE }),
    ).not.toBeInTheDocument()
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
