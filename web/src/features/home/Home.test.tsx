import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { Home } from "@/features/home/Home"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext, usageTotals } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/** `/organizations/me` answering `status`; every other read answers empty. */
async function renderHome(status: number, detail: string) {
  const fetch = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input) => {
      const path = String(input).replace(API_ROOT, "").split("?")[0]
      if (path === "/organizations/me") return json({ detail }, status)
      return json([])
    })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <Home />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
  // The context has answered, so what shows is the answer and not the wait.
  await waitFor(() => {
    expect(
      fetch.mock.calls.some(([input]) =>
        String(input).includes("/organizations/me"),
      ),
    ).toBe(true)
    expect(client.isFetching()).toBe(0)
  })
}

describe("Home", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("reports a context it could not read instead of loading forever", async () => {
    await renderHome(500, "The organization is unavailable")
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The organization is unavailable",
    )
    expect(screen.queryByText("Loading…")).toBeNull()
  })

  it("gives an admin in no workspace the team's Home, which spans the organization", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = String(input).replace(API_ROOT, "").split("?")[0]
      if (path === "/organizations/me") {
        return json(
          organizationContext({
            role: "admin",
            deployment_operator: false,
            workspace_memberships: [],
          }),
        )
      }
      // Every read Home makes answers in its own shape, so nothing it draws
      // depends on which read lands first.
      if (path === "/providers") return json({ providers: [] })
      if (path.endsWith("/usage/summary")) {
        return json({
          totals: usageTotals({ cost: 0, request_count: 0 }),
          series: [],
          by_user: [],
          by_api_key: [],
          by_model: [],
          errors_by_status_code: [],
        })
      }
      if (path.endsWith("/traces/count")) return json({ count: 0 })
      if (
        path === "/organizations/me/members" ||
        path === "/organizations/me/spend-ceilings" ||
        path === "/workspaces"
      ) {
        return json({ data: [], count: 0 })
      }
      return json([])
    })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    await renderWithRouter(
      <QueryClientProvider client={client}>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <Home />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </QueryClientProvider>,
    )
    expect(
      await screen.findByRole("region", { name: "Where it went" }),
    ).toBeInTheDocument()
    expect(screen.queryByText(/not in a workspace yet/)).toBeNull()
  })

  it("leaves a caller with no organization to the shell, with no error", async () => {
    await renderHome(404, "Organization not found")
    expect(await screen.findByText("Loading…")).toBeVisible()
    expect(screen.queryByRole("alert")).toBeNull()
  })
})
