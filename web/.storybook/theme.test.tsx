import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
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

// The decorator reads `context.globals.theme` and nothing else, and renders the
// story as `<Story />`. Narrowing it once says that; asserting a story function
// and a whole `StoryContext` at the call site only says the arguments are not
// what `Decorator` describes.
const decorate = withTheme as unknown as (
  Story: () => ReactNode,
  context: { globals: { theme: string } },
) => ReactNode

function Catalog({ theme }: { theme: string }) {
  return decorate(ThemeConsumer, { globals: { theme } })
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
