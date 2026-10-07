import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react"

/** What a visitor can choose. "Follow the system" is not a choice: see `ThemeProvider`. */
export type ThemePreference = "light" | "dark"

// Exported so `useTheme.test.tsx` can pin it against the pre-paint script in
// `index.html`, which hand-duplicates this key and cannot import it. Renaming it
// here alone would leave that script reading a dead key, and the only symptom
// would be the light flash it exists to prevent, on a dark-mode load.
export const STORAGE_KEY = "otari.dashboard.theme"
// Exported for the same reason as STORAGE_KEY above.
export const DARK_QUERY = "(prefers-color-scheme: dark)"

interface Theme {
  /** What the dashboard is showing right now. */
  resolved: ThemePreference
  setPreference: (preference: ThemePreference) => void
  /** Switch to the other one, which is the only control the product offers. */
  toggle: () => void
}

const Context = createContext<Theme | null>(null)

function isPreference(value: string | null): value is ThemePreference {
  return value === "light" || value === "dark"
}

/**
 * The choice this browser remembers, or `null` when it has made none.
 *
 * `null` also answers a stored `"system"`, which an earlier version wrote and
 * this one no longer offers: a browser holding it was following the operating
 * system, and treating it as nothing stored keeps it doing exactly that until
 * the first click, with no write and no migration.
 */
function readStored(): ThemePreference | null {
  if (typeof window === "undefined") return null
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    return isPreference(stored) ? stored : null
  } catch {
    // Private-mode Safari and a disabled-storage policy both throw. A remembered
    // theme is a convenience; following the system one is no worse than a
    // first visit.
    return null
  }
}

function systemPrefersDark(): boolean {
  if (
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return false
  }
  return window.matchMedia(DARK_QUERY).matches
}

/**
 * The dashboard's light/dark theme.
 *
 * `globals.css` has carried a complete dark token block since the design
 * foundation was rehomed, under `.dark, [data-theme="dark"]`, but nothing ever
 * set the attribute. This is what sets it, on `<html>` so the tokens cover the
 * whole document rather than a subtree.
 *
 * Two states, light and dark, and one control that flips between them. Before
 * anything is chosen the dashboard follows the operating system, so a first
 * visit looks right; the first click stores an explicit choice and the
 * operating system is not consulted again.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [stored, setStored] = useState<ThemePreference | null>(readStored)
  const [systemDark, setSystemDark] = useState<boolean>(systemPrefersDark)

  // Kept subscribed whether or not a choice is stored: it only matters while
  // none is, and subscribing unconditionally keeps this effect free of a
  // dependency that would tear the listener down on the first click.
  useEffect(() => {
    if (
      typeof window === "undefined" ||
      typeof window.matchMedia !== "function"
    )
      return
    const query = window.matchMedia(DARK_QUERY)
    const onChange = (event: MediaQueryListEvent) =>
      setSystemDark(event.matches)
    // Safari below 14 has only the deprecated pair, and the shell already
    // supports that browser for its own media query.
    if (query.addEventListener) {
      query.addEventListener("change", onChange)
      return () => query.removeEventListener("change", onChange)
    }
    query.addListener(onChange)
    return () => query.removeListener(onChange)
  }, [])

  const resolved: ThemePreference = stored ?? (systemDark ? "dark" : "light")

  useEffect(() => {
    const root = document.documentElement
    root.setAttribute("data-theme", resolved)
    // Both spellings, because the `dark:` variant matches either (globals.css).
    root.classList.toggle("dark", resolved === "dark")
    // The tokens only reach what this stylesheet paints. Scrollbars, native
    // checkboxes (Members, Users, Keys and Settings all use one) and form
    // controls are painted by the browser, which follows `color-scheme` alone.
    // `globals.css` declares `light dark` there, meaning "follow the OS", so
    // without this an operator on a light OS who picks Dark gets light native
    // controls on a dark page. Set inline so it wins over that declaration, and
    // in JS rather than in the stylesheet so the rehomed file stays the file
    // otari-ai has (see web/AGENTS.md).
    root.style.colorScheme = resolved
  }, [resolved])

  const setPreference = useCallback((next: ThemePreference) => {
    setStored(next)
    try {
      window.localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // See readStored: the choice still applies, it just is not remembered.
    }
  }, [])

  const toggle = useCallback(
    () => setPreference(resolved === "dark" ? "light" : "dark"),
    [resolved, setPreference],
  )

  const value = useMemo(
    () => ({ resolved, setPreference, toggle }),
    [resolved, setPreference, toggle],
  )

  return <Context.Provider value={value}>{children}</Context.Provider>
}

export function useTheme(): Theme {
  const value = useContext(Context)
  if (!value) {
    throw new Error("useTheme must be used within a ThemeProvider")
  }
  return value
}
