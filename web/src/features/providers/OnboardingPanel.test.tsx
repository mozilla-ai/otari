import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import type { DeploymentBootstrap } from "@/client"
import * as apiClient from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { OnboardingPanel } from "./OnboardingPanel"

function renderPanel(
  deployment: Partial<DeploymentBootstrap>,
  secretKeyConfigured = true,
) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <DeploymentProvider value={bootstrap(deployment)}>
        <OnboardingPanel
          onAddProvider={vi.fn()}
          needsPricing={false}
          onEnablePricing={vi.fn()}
          isEnabling={false}
          secretKeyConfigured={secretKeyConfigured}
        />
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

it("offers otari.ai beside a provider of your own, as the same kind of button", async () => {
  vi.spyOn(apiClient, "apiFetch").mockResolvedValue({ status: "idle" })
  renderPanel({ platform_connect: true })

  const connect = await screen.findByRole("button", {
    name: "Connect otari.ai",
  })
  const own = screen.getByRole("button", { name: "Add your first provider" })
  expect(connect).toBeEnabled()
  expect(own).toBeEnabled()
  // Neither is the page's one accent over the other.
  expect(connect.className).toBe(own.className)
})

it("mentions otari.ai nowhere and reads no connect status where the deployment does not offer it", () => {
  const apiFetch = vi.spyOn(apiClient, "apiFetch")
  renderPanel({ platform_connect: false })

  expect(
    screen.getByRole("button", { name: "Add your first provider" }),
  ).toBeInTheDocument()
  expect(screen.queryByText(/otari\.ai/)).not.toBeInTheDocument()
  expect(apiFetch).not.toHaveBeenCalled()
})

it("says why otari.ai cannot be connected while the server has no secret key", () => {
  const apiFetch = vi.spyOn(apiClient, "apiFetch")
  renderPanel({ platform_connect: true }, false)

  expect(screen.getByText(/OTARI_SECRET_KEY/)).toBeInTheDocument()
  expect(
    screen.queryByRole("button", { name: "Connect otari.ai" }),
  ).not.toBeInTheDocument()
  expect(apiFetch).not.toHaveBeenCalled()
})

it("links the quickstart to the configured documentation, still in a new tab", () => {
  renderPanel({ docs_url: "https://docs.example.com/" })

  const link = screen.getByRole("link", { name: "quickstart" })
  expect(link).toHaveAttribute("href", "https://docs.example.com/")
  expect(link).toHaveAttribute("target", "_blank")
})

it("links the quickstart to the gateway's welcome page where no documentation is configured", () => {
  renderPanel({ docs_url: null })

  const link = screen.getByRole("link", { name: "quickstart" })
  expect(link).toHaveAttribute("href", "/welcome")
  expect(link).toHaveAttribute("target", "_blank")
})
