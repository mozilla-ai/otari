import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useLocation } from "@tanstack/react-router"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { FirstRunStrip } from "@/features/home/FirstRunStrip"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const GET_STARTED = "Get started with Otari"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/** The catalog shape `${API_ROOT}/models` answers, which is the band's gate. */
function modelCatalog(ids: string[]) {
  return {
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      created: 0,
      owned_by: "openai",
    })),
  }
}

interface Bodies {
  /** What GET /hosted-providers lists. */
  hostedProviders?: unknown[]
  logs?: unknown
  providers?: unknown
  /** Answers the providers read with a server error instead of a list. */
  providersFail?: boolean
  context?: Parameters<typeof organizationContext>[0]
  /** Model ids the caller's catalog reports. */
  models?: string[]
  /** Answers the catalog with a server error instead of a list. */
  modelsFail?: boolean
}

// Order matters: /hosted-providers is matched before the bare /providers, and
// /organizations/me on an exact match so it cannot shadow anything under it.
function mockApi(b: Bodies): string[] {
  const requested: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    requested.push(url)
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(organizationContext(b.context))
    }
    if (url.includes(`${API_ROOT}/models`)) {
      if (b.modelsFail) return jsonResponse({ detail: "catalog down" }, 500)
      return jsonResponse(modelCatalog(b.models ?? ["openai:gpt-4o-mini"]))
    }
    if (url.includes(`${API_ROOT}/hosted-providers`)) {
      const hosted = b.hostedProviders ?? []
      return jsonResponse({ data: hosted, count: hosted.length })
    }
    if (url.includes(`${API_ROOT}/providers`)) {
      if (b.providersFail) {
        return jsonResponse({ detail: "providers exploded" }, 500)
      }
      return jsonResponse({
        providers: b.providers ?? [{ provider: "openai" }],
      })
    }
    // The caller's own usage log, which a tenant reads from the organization
    // surface rather than the deployment-wide one.
    if (
      url.includes(`${API_ROOT}/usage`) ||
      url.includes(`${API_ROOT}/organizations/me/usage`)
    ) {
      return jsonResponse(b.logs ?? [])
    }
    return jsonResponse([])
  })
  return requested
}

const asked = (requested: string[], path: string) =>
  requested.some((url) => url.includes(`${API_ROOT}${path}`))

/** Reports where a navigation landed, for the routes `renderStrip` mounts. */
function LocationProbe() {
  const loc = useLocation()
  return <div data-testid="loc">{loc.pathname}</div>
}

async function renderStrip(selected?: string) {
  if (selected) {
    window.localStorage.setItem("otari.dashboard.selectedWorkspace", selected)
  }
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <SelectedWorkspaceProvider>
          <FirstRunStrip />
        </SelectedWorkspaceProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
    {
      routes: [
        { path: "/admin/hosted-providers", element: <LocationProbe /> },
        { path: "/organization/provider-keys", element: <LocationProbe /> },
      ],
    },
  )
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 50))

describe("the operator's first-steps strip", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows a getting-started band and links to providers on a fresh gateway", async () => {
    // The deployment's own credentials: only an operator is sent to Hosted
    // providers, since the tenant band links to the organization's keys.
    mockApi({ providers: [] })
    const user = userEvent.setup()
    await renderStrip()

    expect(await screen.findByText(GET_STARTED)).toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Add your first provider" }),
    )
    expect(await screen.findByTestId("loc")).toHaveTextContent(
      "/admin/hosted-providers",
    )
  })

  it("counts a hosted provider as a configured one", async () => {
    // Hosted providers are not in GET /providers, which lists config.yml
    // instances only, so a gateway whose one provider is hosted is not empty.
    const requested = mockApi({
      providers: [],
      hostedProviders: [{ name: "openai" }],
    })
    await renderStrip()

    await waitFor(() =>
      expect(asked(requested, "/hosted-providers")).toBe(true),
    )
    await settle()
    expect(screen.queryByText(GET_STARTED)).toBeNull()
  })

  it("is not a fresh gateway when it already has usage", async () => {
    // Imported OTLP usage needs no provider config.
    const requested = mockApi({ providers: [], logs: [{ id: "1" }] })
    await renderStrip()

    await waitFor(() => expect(asked(requested, "/usage")).toBe(true))
    await settle()
    expect(screen.queryByText(GET_STARTED)).toBeNull()
  })

  it("says nothing, rather than calling the gateway fresh, when the providers cannot be read", async () => {
    const requested = mockApi({ providersFail: true })
    await renderStrip()

    await waitFor(() => expect(asked(requested, "/providers")).toBe(true))
    await settle()
    expect(screen.queryByText(GET_STARTED)).not.toBeInTheDocument()
    expect(screen.queryByText(/providers exploded/)).not.toBeInTheDocument()
  })
})

describe("the first-steps strip for a caller who does not operate the deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  const WORKSPACE_A = "44444444-4444-4444-4444-444444444444"
  const WORKSPACE_B = "55555555-5555-5555-5555-555555555555"

  const TWO_WORKSPACES = {
    deployment_operator: false,
    workspace_memberships: [
      { workspace_id: WORKSPACE_A, name: "Default Workspace", role: "owner" },
      { workspace_id: WORKSPACE_B, name: "Staging", role: "member" },
    ],
  }

  it("points an admin of a brand-new organization at adding a provider key", async () => {
    mockApi({ context: TWO_WORKSPACES, models: [] })
    const user = userEvent.setup()
    await renderStrip(WORKSPACE_A)

    expect(await screen.findByText(GET_STARTED)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Add a provider key" }))
    expect(await screen.findByTestId("loc")).toHaveTextContent(
      "/organization/provider-keys",
    )
  })

  it("tells a member who to ask, and offers no button they cannot use", async () => {
    mockApi({
      context: { ...TWO_WORKSPACES, role: "member" },
      models: [],
    })
    await renderStrip(WORKSPACE_A)

    expect(await screen.findByText(GET_STARTED)).toBeInTheDocument()
    expect(screen.getByText(/Ask an owner or admin/)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add a provider key" }),
    ).not.toBeInTheDocument()
  })

  it("is absent once there is something to route to", async () => {
    const requested = mockApi({ context: TWO_WORKSPACES })
    await renderStrip(WORKSPACE_A)

    await waitFor(() => expect(asked(requested, "/models")).toBe(true))
    await settle()
    expect(screen.queryByText(GET_STARTED)).not.toBeInTheDocument()
  })

  it("is absent when the account already has traffic", async () => {
    const requested = mockApi({
      context: TWO_WORKSPACES,
      models: [],
      logs: [
        {
          id: "1",
          user_id: null,
          api_key_id: null,
          timestamp: "2026-07-22T00:00:00Z",
          model: "gpt-5.6",
          provider: "openai",
          endpoint: "/v1/chat/completions",
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          cost: 0.5,
          status: "success",
          error_message: null,
          latency_ms: 120,
        },
      ],
    })
    await renderStrip(WORKSPACE_A)

    await waitFor(() =>
      expect(asked(requested, "/organizations/me/usage")).toBe(true),
    )
    await settle()
    expect(screen.queryByText(GET_STARTED)).not.toBeInTheDocument()
  })

  it("stays out of the way when the catalog cannot be read", async () => {
    const requested = mockApi({
      context: TWO_WORKSPACES,
      modelsFail: true,
    })
    await renderStrip(WORKSPACE_A)

    await waitFor(() => expect(asked(requested, "/models")).toBe(true))
    await settle()
    expect(screen.queryByText(GET_STARTED)).not.toBeInTheDocument()
  })
})
