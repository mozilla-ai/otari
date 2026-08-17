import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { HybridLanding } from "@/app/HybridLanding"
import type { DeploymentBootstrap } from "@/client"
import { ApiError, apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

const HYBRID: Partial<DeploymentBootstrap> = {
  deployment_type: "hybrid",
  session_type: "none",
  surfaces: [],
  management_url: "https://otari.ai",
}

function renderLanding(overrides: Partial<DeploymentBootstrap> = {}) {
  return render(
    <AppProviders>
      <DeploymentProvider value={bootstrap({ ...HYBRID, ...overrides })}>
        <HybridLanding />
      </DeploymentProvider>
    </AppProviders>,
  )
}

/** What `/health` answers on a hybrid gateway. */
function health(platformReachable: "yes" | "no", status = "healthy") {
  vi.mocked(apiFetch).mockResolvedValue({
    status,
    mode: "hybrid",
    platform_reachable: platformReachable,
  } as never)
}

describe("HybridLanding", () => {
  beforeEach(() => {
    health("yes")
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("reports the gateway and its control plane as two separate conditions", async () => {
    renderLanding()

    expect(await screen.findByText("Healthy")).toBeInTheDocument()
    expect(screen.getByText("Connected")).toBeInTheDocument()
  })

  it("blames the control plane only when the gateway says so", async () => {
    health("no")
    renderLanding()

    expect(await screen.findByText("Unreachable")).toBeInTheDocument()
    // The gateway itself answered, so it is not the thing that is down.
    expect(screen.getByText("Healthy")).toBeInTheDocument()
  })

  it("leaves the connection unknown when the gateway does not answer", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(0, "Network error: could not reach the gateway."),
    )
    renderLanding()

    expect(await screen.findByText("Not responding")).toBeInTheDocument()
    // Nothing here can see otari.ai except through the gateway, so a dead
    // gateway must not be reported as an unreachable control plane.
    expect(screen.getByText("Unknown")).toBeInTheDocument()
  })

  it("shows the base URL a client is pointed at", async () => {
    renderLanding()

    expect(
      await screen.findByText(`${window.location.origin}/v1`),
    ).toBeInTheDocument()
  })

  it("links out to the configured control plane", async () => {
    renderLanding()

    const link = await screen.findByRole("link", {
      name: "Manage this gateway on otari.ai",
    })
    expect(link).toHaveAttribute("href", "https://otari.ai")
    // An outbound link from a page that knows which gateway this is: opening it
    // in a new tab must not hand otari.ai this deployment's URL, nor a handle on
    // this window.
    expect(link).toHaveAttribute("rel", "noreferrer")
    expect(link).toHaveAttribute("target", "_blank")
  })

  it("offers no link when no control plane was configured", async () => {
    renderLanding({ management_url: null })

    expect(await screen.findByText("Healthy")).toBeInTheDocument()
    expect(screen.queryByRole("link")).toBeNull()
  })

  it("reads nothing but the public health endpoint", async () => {
    renderLanding()
    await screen.findByText("Healthy")

    // The management API does not exist on a hybrid gateway: every /v1/ path the
    // dashboard knows answers 404 there. A page that asked anyway would render a
    // wall of errors, and a page that asked for a *credential* is the failure the
    // deployment contract exists to prevent. So the whole page is one public read.
    const paths = vi.mocked(apiFetch).mock.calls.map(([path]) => path)
    expect(paths.length).toBeGreaterThan(0)
    expect(new Set(paths)).toEqual(new Set(["/health"]))
  })

  it("exposes no management surface", async () => {
    const { container } = renderLanding()
    await screen.findByText("Healthy")

    // Not a shell with its pages hidden: there is no navigation at all, and the
    // only link leaves for otari.ai rather than entering a local route.
    expect(screen.queryByRole("navigation")).toBeNull()
    const hrefs = [...container.querySelectorAll("a")].map((a) =>
      a.getAttribute("href"),
    )
    expect(hrefs).toEqual(["https://otari.ai"])
  })

  it("renders no bootstrap field it was not written to render", async () => {
    // The bootstrap carries no secret (asserted in test_deployment_bootstrap.py),
    // and this is the client half of that: the page reads one field, the link
    // target, rather than painting whatever it was handed. So a field added to
    // the contract later cannot reach the screen without someone choosing to put
    // it there.
    render(
      <AppProviders>
        <DeploymentProvider
          value={
            {
              ...bootstrap(HYBRID),
              platform_token: "gw_secret_token",
            } as DeploymentBootstrap
          }
        >
          <HybridLanding />
        </DeploymentProvider>
      </AppProviders>,
    )
    await screen.findByText("Healthy")

    expect(document.body.textContent).not.toContain("gw_secret_token")
  })
})
