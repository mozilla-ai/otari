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

// The layout fact being pinned: the appearance toggle's box is 44x44 at every
// width, with no `md:` step down, because the header it sits in is `min-h-14`
// and has the room. jsdom performs no layout, so the classes that cause the box
// are the only thing a unit test can see (#1336).
it("keeps the appearance toggle at the 44px touch floor", () => {
  localStorage.setItem(STORAGE_KEY, "system")
  render(
    <ThemeProvider>
      <LoginPageShell>
        <h1>Sign in</h1>
      </LoginPageShell>
    </ThemeProvider>,
  )
  const toggle = screen.getByRole("button", {
    name: "Appearance: system. Switch to light.",
  })
  expect(toggle).toHaveClass("min-h-11", "min-w-11")
  expect(toggle.className).not.toContain("md:min-h-")
})
