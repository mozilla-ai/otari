import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

// The page's empty state. The table and the create
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

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <RoutingPage />
      </QueryClientProvider>
    </DeploymentProvider>,
    { wrapper: withRouter({ url: "/" }) },
  )
}

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
})
