import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, expect, it, vi } from "vitest"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { LoginPageShell } from "./LoginPageShell"

vi.mock("./background/LoginBackground", () => ({
  LoginBackground: ({ paused }: { paused: boolean }) => (
    <span>{paused ? "Motion paused" : "Motion playing"}</span>
  ),
}))
afterEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
  document.documentElement.removeAttribute("data-theme")
  document.documentElement.classList.remove("dark")
  document.documentElement.style.removeProperty("color-scheme")
})

it("offers a keyboard-accessible pause and theme switch outside the form", async () => {
  localStorage.setItem("otari.dashboard.theme", "light")
  const user = userEvent.setup()
  render(
    <ThemeProvider>
      <LoginPageShell>
        <h1>Sign in</h1>
      </LoginPageShell>
    </ThemeProvider>,
  )
  await user.click(
    screen.getByRole("button", { name: "Pause background animation" }),
  )
  expect(screen.getByText("Motion paused")).toBeInTheDocument()
  await user.click(
    screen.getByRole("button", { name: "Play background animation" }),
  )
  expect(screen.getByText("Motion playing")).toBeInTheDocument()
  await user.click(screen.getByRole("button", { name: "Use dark theme" }))
  expect(document.documentElement).toHaveAttribute("data-theme", "dark")
  expect(screen.getByRole("main")).toContainElement(
    screen.getByRole("heading", { name: "Sign in" }),
  )
})
