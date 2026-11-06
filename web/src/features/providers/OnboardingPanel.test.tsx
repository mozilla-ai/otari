import { render, screen } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import type { DeploymentBootstrap } from "@/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { OnboardingPanel } from "./OnboardingPanel"

function renderPanel(deployment: Partial<DeploymentBootstrap>) {
  render(
    <DeploymentProvider value={bootstrap(deployment)}>
      <OnboardingPanel
        onAddProvider={vi.fn()}
        needsPricing={false}
        onEnablePricing={vi.fn()}
        isEnabling={false}
        secretKeyConfigured
      />
    </DeploymentProvider>,
  )
}

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
