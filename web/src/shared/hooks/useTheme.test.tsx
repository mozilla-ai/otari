import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ThemeProvider, useTheme } from "@/shared/hooks/useTheme"

function mockPrefersDark(dark: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn().mockReturnValue({
      matches: dark,
      addEventListener: () => {},
      removeEventListener: () => {},
    }),
  )
}

function Probe() {
  const { preference, resolved, setPreference } = useTheme()
  return (
    <div>
      <span data-testid="pref">{preference}</span>
      <span data-testid="resolved">{resolved}</span>
      <button type="button" onClick={() => setPreference("dark")}>
        dark
      </button>
      <button type="button" onClick={() => setPreference("system")}>
        system
      </button>
    </div>
  )
}

describe("useTheme", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    window.localStorage.clear()
    document.documentElement.removeAttribute("data-theme")
    document.documentElement.classList.remove("dark")
  })

  it("follows the system preference by default", () => {
    mockPrefersDark(true)
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    )

    expect(screen.getByTestId("pref")).toHaveTextContent("system")
    expect(screen.getByTestId("resolved")).toHaveTextContent("dark")
  })

  it("puts the resolved theme on the document, which is what the tokens key off", async () => {
    mockPrefersDark(false)
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    )
    expect(document.documentElement).toHaveAttribute("data-theme", "light")

    await userEvent.click(screen.getByRole("button", { name: "dark" }))

    // Both spellings, because globals.css matches either.
    expect(document.documentElement).toHaveAttribute("data-theme", "dark")
    expect(document.documentElement.classList.contains("dark")).toBe(true)
  })

  it("keeps an explicit choice apart from the system one", async () => {
    mockPrefersDark(true)
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    )

    await userEvent.click(screen.getByRole("button", { name: "system" }))
    // "system" is a preference of its own, not a resolved light/dark: it has to
    // survive so the theme keeps following the OS afterwards.
    expect(screen.getByTestId("pref")).toHaveTextContent("system")
    expect(screen.getByTestId("resolved")).toHaveTextContent("dark")
  })
})
