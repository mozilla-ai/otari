import { readFileSync } from "node:fs"
import { join } from "node:path"

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  DARK_QUERY,
  STORAGE_KEY,
  THEME_PREFERENCES,
  ThemeProvider,
  useTheme,
} from "@/shared/hooks/useTheme"

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

describe("the pre-paint script in index.html", () => {
  // It cannot import from this module: it runs before the bundle exists, which
  // is the whole point of it. So the key and the resolution are hand-copied,
  // and the copy can only be kept honest from here. A rename that misses the
  // script leaves it reading nothing, and nothing fails except a dark-mode
  // operator seeing a white flash on every load.
  // From the vitest root (web/), matching `routes.test.ts` and
  // `architecture.test.ts`; `import.meta.url` is not a file URL under jsdom.
  const html = readFileSync(join(process.cwd(), "index.html"), "utf8")

  it("reads the key this module writes", () => {
    expect(html).toContain(`"${STORAGE_KEY}"`)
  })

  it("resolves the same three states", () => {
    expect(html).toContain(DARK_QUERY)
    for (const preference of THEME_PREFERENCES) {
      if (preference === "system") continue
      expect(html).toContain(`"${preference}"`)
    }
  })
})
