import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query"
import {
  act,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { createElement, type ReactNode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TelemetryIdentity } from "@/app/TelemetryIdentity"
import type { OrganizationContext } from "@/client"
import { PasswordCard } from "@/features/account/PasswordCard"
import { ProfileCard } from "@/features/account/ProfileCard"
import { AuthProvider, useAuth } from "@/features/auth/AuthContext"
import { SetupGuide } from "@/features/onboarding/SetupGuide"
import { KEPT_ANSWER_HOURS, keptAnswer } from "@/shared/api/keptAnswers"
import {
  deploymentOperatorAnswer,
  useKeepOrganizationContext,
  useOrganizationContext,
  useSwitchOrganization,
} from "@/shared/api/organizations"
import {
  type RequestPolicy,
  SAME_ORIGIN_POLICY,
} from "@/shared/api/requestPolicy"
import {
  SelectedWorkspaceProvider,
  useSelectedWorkspace,
} from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { identify, resetTelemetrySpy } from "@/tests/telemetry"

const policy = vi.hoisted(() => ({ current: undefined as unknown }))

vi.mock("@/shared/api/overlayRequestPolicy", () => ({
  prepareRequests: async () => {},
  requestPolicy: () => policy.current,
}))

vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

const KEPT = "otari.dashboard.organizationContext"
const SESSION = "otari.dashboard.hasSession"
const ALPHA = "44444444-4444-4444-4444-444444444444"
const BETA = "55555555-5555-5555-5555-555555555555"

const KEEPING: RequestPolicy = {
  origin: "",
  credentials: "same-origin",
  keptAnswers: { bootstrap: false, organizationContext: true },
}

function named(name: string, overrides: Partial<OrganizationContext> = {}) {
  return organizationContext({
    deployment_operator: false,
    organization: { ...organizationContext().organization, name },
    caller: {
      user_id: "66666666-6666-6666-6666-666666666666",
      email: "ada@example.com",
      full_name: "Ada Lovelace",
      has_password: true,
      claims_deployment: false,
    },
    ...overrides,
  })
}

const kept = named("Kept", {
  workspace_memberships: [
    { workspace_id: ALPHA, name: "Alpha", role: "owner" },
  ],
})
const current = named("Current", {
  workspace_memberships: [{ workspace_id: BETA, name: "Beta", role: "owner" }],
})

function keep(context: OrganizationContext = kept, origin = "") {
  keptAnswer<OrganizationContext>(KEPT, origin).write(context)
}

function createClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } })
}

function createWrapper(client: QueryClient = createClient()) {
  return ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client }, children)
}

/** `/organizations/me` held open until the test answers it. */
function holdContext() {
  let answer: (response: Response) => void = () => {}
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) =>
    String(input).endsWith("/organizations/me")
      ? new Promise((resolve) => {
          answer = resolve
        })
      : Promise.resolve(Response.json({})),
  )
  return {
    fetchMock,
    answer: (response: Response) => act(() => answer(response)),
  }
}

function marks(): string[] {
  return performance
    .getEntriesByType("mark")
    .map((entry) => entry.name)
    .filter((name) => name.startsWith("ctx:"))
}

beforeEach(() => {
  policy.current = KEEPING
  window.localStorage.setItem(SESSION, "1")
  resetTelemetrySpy()
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
  window.localStorage.clear()
  performance.clearMarks()
})

describe("a request policy that keeps no organization context", () => {
  it("shows none, writes none, and forgets one an earlier build kept", async () => {
    policy.current = SAME_ORIGIN_POLICY
    keep()
    const { answer } = holdContext()

    const { result } = renderHook(
      () => {
        useKeepOrganizationContext()
        return useOrganizationContext()
      },
      { wrapper: createWrapper() },
    )

    expect(result.current.data).toBeUndefined()
    await waitFor(() => expect(window.localStorage.getItem(KEPT)).toBeNull())
    await answer(Response.json(current))
    await waitFor(() =>
      expect(result.current.data?.organization.name).toBe("Current"),
    )
    expect(window.localStorage.getItem(KEPT)).toBeNull()
    expect(marks()).toEqual(["ctx:kept:off", "ctx:current"])
  })
})

describe("a request policy that keeps the organization context", () => {
  it("stands in while the current context loads, then gives way to it", async () => {
    keep()
    const { answer } = holdContext()

    const { result } = renderHook(
      () => {
        useKeepOrganizationContext()
        return useOrganizationContext()
      },
      { wrapper: createWrapper() },
    )

    expect(result.current.data?.organization.name).toBe("Kept")
    expect(result.current.isPlaceholderData).toBe(true)
    // Settled at once, so nothing that waits on the answer waits a round trip.
    expect(deploymentOperatorAnswer(result.current)).toBe("not-operator")

    await answer(Response.json(current))
    await waitFor(() =>
      expect(result.current.data?.organization.name).toBe("Current"),
    )
    expect(result.current.isPlaceholderData).toBe(false)
    expect(marks()).toEqual(["ctx:kept:hit", "ctx:current"])
  })

  it("marks a miss when nothing was kept", async () => {
    const { answer } = holdContext()

    renderHook(() => useKeepOrganizationContext(), {
      wrapper: createWrapper(),
    })
    await answer(Response.json(current))

    await waitFor(() =>
      expect(marks()).toEqual(["ctx:kept:miss", "ctx:current"]),
    )
  })

  it("keeps what starts the reads and picks the page, and nothing about the person", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      Response.json(String(input).endsWith("/switch") ? kept : current),
    )

    const { result } = renderHook(
      () => {
        useKeepOrganizationContext()
        return useSwitchOrganization()
      },
      { wrapper: createWrapper() },
    )

    await waitFor(() =>
      expect(window.localStorage.getItem(KEPT)).not.toBeNull(),
    )
    const stored = window.localStorage.getItem(KEPT) ?? ""
    expect(JSON.parse(stored)).toEqual({
      origin: "",
      keptAt: expect.any(Number),
      value: {
        organization: {
          id: current.organization.id,
          name: "Current",
          slug: current.organization.slug,
          created_at: current.organization.created_at,
        },
        organization_member_id: current.organization_member_id,
        role: current.role,
        status: current.status,
        deployment_operator: false,
        provider_key_encryption_available: true,
        workspace_memberships: [
          { workspace_id: BETA, name: "Beta", role: "owner" },
        ],
      },
    })
    expect(stored).not.toContain("ada@example.com")
    expect(stored).not.toContain("Ada Lovelace")
    expect(stored).not.toContain("66666666")

    // A switch writes the cache, and the next load starts from it.
    await act(() => result.current.mutateAsync("org-there"))
    await waitFor(() =>
      expect(
        JSON.parse(window.localStorage.getItem(KEPT) ?? "null")?.value
          .organization.name,
      ).toBe("Kept"),
    )
  })

  it("is not offered without a session, to a read that is disabled, or to another API origin", () => {
    holdContext()
    keep()
    window.localStorage.removeItem(SESSION)
    expect(
      renderHook(() => useOrganizationContext(), { wrapper: createWrapper() })
        .result.current.data,
    ).toBeUndefined()

    window.localStorage.setItem(SESSION, "1")
    expect(
      renderHook(() => useOrganizationContext(false), {
        wrapper: createWrapper(),
      }).result.current.data,
    ).toBeUndefined()

    keep(kept, "https://eu.api.otari.test")
    expect(
      renderHook(() => useOrganizationContext(), { wrapper: createWrapper() })
        .result.current.data,
    ).toBeUndefined()
  })

  it("is not offered once it is older than a dashboard session lives", () => {
    vi.useFakeTimers({ toFake: ["Date"] })
    holdContext()
    keep()
    vi.advanceTimersByTime(KEPT_ANSWER_HOURS * 60 * 60 * 1000)

    const { result } = renderHook(() => useOrganizationContext(), {
      wrapper: createWrapper(),
    })

    expect(result.current.data).toBeUndefined()
  })

  it("reports a removed member's 404 instead of the kept context, and forgets it", async () => {
    keep()
    const { fetchMock, answer } = holdContext()

    const { result } = renderHook(() => useOrganizationContext(), {
      wrapper: createWrapper(),
    })
    expect(result.current.data?.organization.name).toBe("Kept")

    await answer(
      Response.json({ detail: "No active membership" }, { status: 404 }),
    )
    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(result.current.data).toBeUndefined()
    expect(deploymentOperatorAnswer(result.current)).toBe("no-organization")
    expect(window.localStorage.getItem(KEPT)).toBeNull()

    fetchMock.mockImplementation(() => new Promise(() => {}))
    act(() => {
      void result.current.refetch()
    })
    await waitFor(() => expect(result.current.isFetching).toBe(true))
    expect(result.current.data).toBeUndefined()
  })

  it("forgets it when the read fails", async () => {
    keep()
    const { answer } = holdContext()

    const { result } = renderHook(() => useOrganizationContext(), {
      wrapper: createWrapper(),
    })
    await answer(Response.json({ detail: "boom" }, { status: 503 }))

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(window.localStorage.getItem(KEPT)).toBeNull()
  })

  it("identifies nobody to telemetry until the current answer arrives", async () => {
    keep()
    const { answer } = holdContext()

    render(<TelemetryIdentity />, { wrapper: createWrapper() })
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(identify).not.toHaveBeenCalled()

    await answer(Response.json(current))
    await waitFor(() =>
      expect(identify).toHaveBeenCalledWith(
        expect.objectContaining({ organizationName: "Current" }),
      ),
    )
    expect(identify).toHaveBeenCalledOnce()
  })

  it("waits for the current answer to show or edit who is signed in, which a kept context does not carry", async () => {
    keep()
    const { answer } = holdContext()

    render(
      <DeploymentProvider value={bootstrap()}>
        <ProfileCard />
        <PasswordCard />
      </DeploymentProvider>,
      { wrapper: createWrapper() },
    )

    expect(screen.getByLabelText("Loading your name")).toBeInTheDocument()
    expect(
      screen.getByLabelText("Loading your sign-in details"),
    ).toBeInTheDocument()
    expect(
      screen.queryByText(/Who is signed in could not be read/),
    ).not.toBeInTheDocument()

    await answer(Response.json(current))
    expect(await screen.findByDisplayValue("Ada Lovelace")).toBeInTheDocument()
  })

  it("selects from the kept workspaces but remembers no selection until the current answer arrives", async () => {
    keep()
    window.localStorage.setItem("otari.dashboard.selectedWorkspace", BETA)
    const { answer } = holdContext()
    function Probe() {
      const { selected, isConfirmed } = useSelectedWorkspace()
      return (
        <span data-testid="selected">
          {selected?.name ?? "none"} {String(isConfirmed)}
        </span>
      )
    }

    render(
      <SelectedWorkspaceProvider>
        <Probe />
      </SelectedWorkspaceProvider>,
      { wrapper: createWrapper() },
    )

    // The kept context lists only Alpha, which a write would repair the
    // stored Beta onto.
    expect(screen.getByTestId("selected")).toHaveTextContent("Alpha false")
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(
      window.localStorage.getItem("otari.dashboard.selectedWorkspace"),
    ).toBe(BETA)

    await answer(Response.json(current))
    await waitFor(() =>
      expect(screen.getByTestId("selected")).toHaveTextContent("Beta true"),
    )
  })

  it("does not ask to set up a kept workspace before the current answer arrives", async () => {
    keep()
    const { fetchMock, answer } = holdContext()

    render(
      <SelectedWorkspaceProvider>
        <SetupGuide canServeRequests />
      </SelectedWorkspaceProvider>,
      { wrapper: createWrapper() },
    )
    await new Promise((resolve) => setTimeout(resolve, 20))
    const activationReads = () =>
      fetchMock.mock.calls.filter(([input]) =>
        String(input).includes("/activation"),
      )
    expect(activationReads()).toEqual([])

    await answer(Response.json(current))
    await waitFor(() =>
      expect(String(activationReads()[0]?.[0])).toContain(
        `/workspaces/${BETA}/activation`,
      ),
    )
  })

  it("holds back the kept workspaces' queries when the caller switches before the current answer", async () => {
    keep()
    const client = createClient()
    vi.spyOn(globalThis, "fetch").mockImplementation((input) =>
      String(input).endsWith("/organizations/me")
        ? new Promise(() => {})
        : Promise.resolve(Response.json(current)),
    )
    const reads = { left: 0 }
    const { result } = renderHook(
      () => {
        useOrganizationContext()
        useQuery({
          queryKey: ["left", ALPHA],
          // `workspaceMeta(ALPHA)`, which only `shared/api` may import.
          meta: { workspaceId: ALPHA },
          queryFn: () => ++reads.left,
        })
        return useSwitchOrganization()
      },
      { wrapper: createWrapper(client) },
    )
    await waitFor(() => expect(reads.left).toBe(1))

    await act(() => result.current.mutateAsync("org-there"))

    expect(reads.left).toBe(1)
    expect(client.getQueryState(["left", ALPHA])?.isInvalidated).toBe(true)
  })
})

describe("signing in and out", () => {
  function SessionHarness() {
    const { isAuthenticated, login, logout } = useAuth()
    return isAuthenticated ? (
      <button type="button" onClick={() => void logout()}>
        Sign out
      </button>
    ) : (
      <button type="button" onClick={login}>
        Sign in
      </button>
    )
  }

  it("forgets the kept context, so it never reaches whoever signs in next", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(null, { status: 204 }),
    )
    keep()
    const user = userEvent.setup()
    render(
      <AuthProvider>
        <SessionHarness />
      </AuthProvider>,
      { wrapper: createWrapper() },
    )

    await user.click(screen.getByRole("button", { name: "Sign out" }))
    expect(window.localStorage.getItem(KEPT)).toBeNull()

    keep()
    await user.click(screen.getByRole("button", { name: "Sign in" }))
    expect(window.localStorage.getItem(KEPT)).toBeNull()
  })
})
