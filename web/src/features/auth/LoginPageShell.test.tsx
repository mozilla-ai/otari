import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, expect, it, vi } from "vitest"
import { STORAGE_KEY, ThemeProvider } from "@/shared/hooks/useTheme"
import { LoginPageShell } from "./LoginPageShell"

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
  render(
    <ThemeProvider>
      <LoginPageShell>
        <h1>Sign in</h1>
      </LoginPageShell>
    </ThemeProvider>,
  )
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
