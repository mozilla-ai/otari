import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import App from "@/app/App"
import { Provider } from "@/app/provider"
import { apiFetch } from "@/shared/api/client"
import { bootstrap } from "@/tests/fixtures"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

vi.mock("@/features/overview/OverviewPage", async () => {
  await new Promise((resolve) => window.setTimeout(resolve, 20))
  return { OverviewIndex: () => <div>Lazy overview</div> }
})

function renderApp(deployment: Parameters<typeof App>[0]["bootstrap"]) {
  return render(
    <Provider>
      <App bootstrap={deployment} />
    </Provider>,
  )
}

describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
    window.location.hash = ""
  })

  it("shows a loading state while the current route loads", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
    vi.mocked(apiFetch).mockImplementation(async (path) => {
      if (path === "/dashboard-build.json") {
        return { build: "test-build" } as never
      }
      if (path === "/v1/settings") {
        return { default_pricing: true, require_pricing: false } as never
      }
      return [] as never
    })

    renderApp(bootstrap())

    expect(screen.getByRole("status")).toHaveTextContent("Loading page…")
    expect(await screen.findByText("Lazy overview")).toBeInTheDocument()
  })

  it("asks a local-operator deployment to sign in", () => {
    // No stored session marker, so the shell is not reachable yet.
    renderApp(bootstrap())

    expect(
      screen.getByRole("heading", { name: "Otari Dashboard" }),
    ).toBeInTheDocument()
  })

  it("renders the data-plane landing page for a hybrid gateway", () => {
    // Signed in locally, which must not matter: a hybrid gateway issues no
    // management session, so there is no dashboard here to reach.
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
    // The landing page's one read. HybridLanding.test.tsx covers what it renders
    // from the answer; this test is about which root the shell picks.
    vi.mocked(apiFetch).mockResolvedValue({
      status: "healthy",
      mode: "hybrid",
      platform_reachable: "yes",
    } as never)

    renderApp(
      bootstrap({
        deployment_type: "hybrid",
        session_type: "none",
        surfaces: [],
        management_url: "https://otari.ai",
      }),
    )

    expect(
      screen.getByRole("heading", { name: "Otari gateway" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Manage this gateway on otari.ai" }),
    ).toHaveAttribute("href", "https://otari.ai")
    // The management shell is not merely hidden behind a sign-in here.
    expect(screen.queryByRole("navigation")).toBeNull()
  })

  it("says so when the deployment context could not be read", () => {
    // main.tsx passes null when /v1/bootstrap did not answer. Assuming a
    // deployment and rendering its dashboard is the failure to avoid.
    renderApp(null)

    expect(screen.getByRole("alert")).toHaveTextContent(
      /does not know what it is connected to/,
    )
    expect(
      screen.queryByRole("heading", { name: "Otari Dashboard" }),
    ).toBeNull()
  })
})
