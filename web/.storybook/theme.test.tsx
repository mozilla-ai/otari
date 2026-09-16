import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { Decorator } from "@storybook/react-vite"
import { afterEach, expect, it } from "vitest"
import { useTheme } from "@/shared/hooks/useTheme"
import { withTheme } from "./theme"

function ThemeConsumer() {
  const { preference, resolved, setPreference } = useTheme()
  return (
    <button type="button" onClick={() => setPreference("light")}>
      {preference}: {resolved}
    </button>
  )
}

function Catalog({ theme }: { theme: string }) {
  return withTheme(
    ThemeConsumer as Parameters<Decorator>[0],
    { globals: { theme } } as Parameters<Decorator>[1],
  )
}

afterEach(() => {
  localStorage.clear()
  document.documentElement.removeAttribute("data-theme")
  document.documentElement.classList.remove("dark")
  document.documentElement.style.removeProperty("color-scheme")
})

it("provides theme context and follows toolbar changes over a stored preference", async () => {
  localStorage.setItem("otari.dashboard.theme", "dark")
  const { rerender } = render(<Catalog theme="light" />)
  expect(await screen.findByRole("button", { name: "light: light" })).toBeVisible()

  rerender(<Catalog theme="dark" />)
  expect(await screen.findByRole("button", { name: "dark: dark" })).toBeVisible()
  await waitFor(() => {
    expect(document.documentElement).toHaveAttribute("data-theme", "dark")
    expect(document.documentElement).toHaveClass("dark")
    expect(document.documentElement.style.colorScheme).toBe("dark")
  })

  await userEvent.setup().click(screen.getByRole("button"))
  expect(await screen.findByRole("button", { name: "light: light" })).toBeVisible()
  await waitFor(() => {
    expect(document.documentElement).toHaveAttribute("data-theme", "light")
    expect(document.documentElement).not.toHaveClass("dark")
    expect(document.documentElement.style.colorScheme).toBe("light")
  })
})
