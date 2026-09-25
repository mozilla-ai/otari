import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { DeploymentBootstrap } from "@/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { STORAGE_KEY, ThemeProvider } from "@/shared/hooks/useTheme"
import { bootstrap } from "@/tests/fixtures"
import { LoginPageShell } from "./LoginPageShell"

function renderShell(
  children: ReactNode = <h1>Sign in</h1>,
  deployment: Partial<DeploymentBootstrap> = {},
) {
  return render(
    <DeploymentProvider value={bootstrap(deployment)}>
      <ThemeProvider>
        <LoginPageShell>{children}</LoginPageShell>
      </ThemeProvider>
    </DeploymentProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
  document.documentElement.removeAttribute("data-theme")
  document.documentElement.classList.remove("dark")
  document.documentElement.style.removeProperty("color-scheme")
})

it("cycles through light, dark, and system without an animation control", async () => {
  localStorage.setItem(STORAGE_KEY, "system")
  const user = userEvent.setup()
  renderShell()
  expect(
    screen.queryByRole("button", { name: /background animation/ }),
  ).not.toBeInTheDocument()
  for (const [current, next] of [
    ["system", "light"],
    ["light", "dark"],
    ["dark", "system"],
  ]) {
    await user.click(
      screen.getByRole("button", {
        name: `Appearance: ${current}. Switch to ${next}.`,
      }),
    )
    expect(localStorage.getItem(STORAGE_KEY)).toBe(next)
  }
  expect(screen.getByRole("main")).toContainElement(
    screen.getByRole("heading", { name: "Sign in" }),
  )
})

// The layout fact being pinned: the appearance toggle's box is 44x44 at every
// width, with no `md:` step down, because the header it sits in is `min-h-14`
// and has the room. jsdom performs no layout, so the classes that cause the box
// are the only thing a unit test can see (#1336).
it("keeps the appearance toggle at the 44px touch floor", () => {
  localStorage.setItem(STORAGE_KEY, "system")
  renderShell()
  const toggle = screen.getByRole("button", {
    name: "Appearance: system. Switch to light.",
  })
  expect(toggle).toHaveClass("min-h-11", "min-w-11")
  expect(toggle.className).not.toContain("md:min-h-")
})

describe("the header's logo", () => {
  it("leads to the deployment's website where one is named", () => {
    renderShell(undefined, {
      site_url: "https://otari.ai/",
      public_catalog: true,
    })
    const link = screen.getByRole("link", { name: "Otari home" })
    expect(link).toHaveAttribute("href", "https://otari.ai/")
    expect(link).not.toHaveAttribute("target")
    // The 44px touch floor; jsdom computes no height, so the class is the
    // only evidence a unit test has.
    expect(link).toHaveClass("min-h-11")
  })

  it("leads to the public catalog when there is no website", () => {
    renderShell(undefined, { site_url: null, public_catalog: true })
    expect(screen.getByRole("link", { name: "Otari home" })).toHaveAttribute(
      "href",
      "#/models",
    )
  })

  // The only other page would be the sign-in screen itself.
  it("stays unlinked with neither a website nor a catalog", () => {
    renderShell(undefined, { site_url: null, public_catalog: false })
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
    expect(screen.getByText("Otari")).toBeInTheDocument()
  })
})
