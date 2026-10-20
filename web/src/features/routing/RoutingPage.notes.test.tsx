import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap } from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  HOSTED_SURFACES,
  organizationContext,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

// The page's empty state and the deployment note. The table and the create
// dialog are covered in RoutingPage.test.tsx and RoutingPage.create.test.tsx.

function mockEmptyApi() {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    const body = url.endsWith(`${API_ROOT}/organizations/me`)
      ? organizationContext()
      : []
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
}

function renderPage(deployment: Partial<DeploymentBootstrap> = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap(deployment)}>
      <QueryClientProvider client={client}>
        <RoutingPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { wrapper: withRouter({ url: "/" }) },
  )
}

const NOTE = /Policies are a standalone feature/

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage empty state", () => {
  it("leads with a fallback, and says a catalog ID already falls back", async () => {
    mockEmptyApi()
    const user = userEvent.setup()
    renderPage()

    const heading = await screen.findByRole("heading", {
      name: "No routing policies yet",
    })
    const empty = heading.closest("div")!.parentElement!
    expect(empty).toHaveTextContent(
      "A policy gives callers one model name with backups behind it",
    )
    expect(
      within(empty)
        .getAllByRole("button")
        .map((button) => button.textContent),
    ).toEqual(["Create fallback", "Other policy types"])
    expect(
      within(empty).getByRole("link", { name: "Browse models" }),
    ).toHaveAttribute("href", "/models")
    expect(
      within(empty).getByRole("link", { name: /read how/ }),
    ).toHaveAttribute(
      "href",
      expect.stringContaining("docs/models.md#catalog-spellings"),
    )

    await user.click(
      within(empty).getByRole("button", { name: "Create fallback" }),
    )
    expect(await screen.findByRole("dialog")).toHaveAccessibleName(
      "New fallback",
    )
  })

  it("leaves the catalog tip off a deployment that compiles no plan itself", async () => {
    mockEmptyApi()
    renderPage({
      deployment_type: "hosted",
      surfaces: [...HOSTED_SURFACES],
    })

    await screen.findByRole("heading", { name: "No routing policies yet" })
    expect(screen.queryByText(/falls back/)).not.toBeInTheDocument()
  })
})

describe("RoutingPage deployment note", () => {
  it("says nothing on a standalone gateway, which honors its policies", async () => {
    mockEmptyApi()
    renderPage()

    await screen.findByRole("heading", { name: "No routing policies yet" })
    expect(screen.queryByText(NOTE)).not.toBeInTheDocument()
  })

  it("says a hosted deployment's policies are not where its plans come from", async () => {
    mockEmptyApi()
    renderPage({
      deployment_type: "hosted",
      surfaces: [...HOSTED_SURFACES],
    })

    expect(await screen.findByText(NOTE)).toHaveTextContent(
      "A hybrid gateway receives its attempt plan from the connected control plane.",
    )
  })

  it("says the same on a hybrid gateway", async () => {
    mockEmptyApi()
    renderPage({ deployment_type: "hybrid", surfaces: [] })

    expect(await screen.findByText(NOTE)).toBeInTheDocument()
  })
})
