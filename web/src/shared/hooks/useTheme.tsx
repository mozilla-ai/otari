import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react"

export const THEME_PREFERENCES = ["system", "light", "dark"] as const
export type ThemePreference = (typeof THEME_PREFERENCES)[number]

const STORAGE_KEY = "otari.dashboard.theme"
const DARK_QUERY = "(prefers-color-scheme: dark)"

interface Theme {
  /** What the operator chose, which may be "system". */
  preference: ThemePreference
  /** What that resolves to right now, which never is. */
  resolved: "light" | "dark"
  setPreference: (preference: ThemePreference) => void
}

const Context = createContext<Theme | null>(null)

function isPreference(value: string | null): value is ThemePreference {
  return (THEME_PREFERENCES as readonly string[]).includes(value ?? "")
}

function readStored(): ThemePreference {
  if (typeof window === "undefined") return "system"
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    return isPreference(stored) ? stored : "system"
  } catch {
    // Private-mode Safari and a disabled-storage policy both throw. A remembered
    // theme is a convenience; falling back to the system one is no worse than a
    // first visit.
    return "system"
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
 * The dashboard's light/dark preference.
 *
 * `globals.css` has carried a complete dark token block since the design
 * foundation was rehomed, under `.dark, [data-theme="dark"]`, but nothing ever
 * set the attribute. This is what sets it, on `<html>` so the tokens cover the
 * whole document rather than a subtree.
 *
 * Three states, not two: "system" is its own preference and keeps following the
 * OS after the fact, which a resolved light/dark pair cannot express.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, setStored] = useState<ThemePreference>(readStored)
  const [systemDark, setSystemDark] = useState<boolean>(systemPrefersDark)

  // Kept subscribed whatever the preference is, so switching back to "system"
  // is already correct rather than correct at the next OS change.
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

  const resolved =
    preference === "system" ? (systemDark ? "dark" : "light") : preference

  useEffect(() => {
    const root = document.documentElement
    root.setAttribute("data-theme", resolved)
    // Both spellings, because the `dark:` variant matches either (globals.css).
    root.classList.toggle("dark", resolved === "dark")
  }, [resolved])

  const setPreference = useCallback((next: ThemePreference) => {
    setStored(next)
    try {
      window.localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // See readStored: the choice still applies, it just is not remembered.
    }
  }, [])

  const value = useMemo(
    () => ({ preference, resolved, setPreference }),
    [preference, resolved, setPreference],
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
