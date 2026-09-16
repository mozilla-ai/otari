import { type ReactNode, useEffect } from "react"
import type { Decorator } from "@storybook/react-vite"

import { THEME_PREFERENCES, ThemeProvider, useTheme } from "@/shared/hooks/useTheme"

/**
 * The names come from `THEME_PREFERENCES` rather than being restated, minus
 * "system": a catalog exists to show both themes deliberately, and "whatever
 * this laptop is set to" is not a case worth a toolbar entry.
 */
const THEMES = THEME_PREFERENCES.filter((preference) => preference !== "system")

export const themeGlobalType = {
  theme: {
    description: "Design-token theme",
    defaultValue: "light",
    toolbar: {
      title: "Theme",
      icon: "circlehollow",
      items: THEMES.map((value) => ({
        value,
        title: value === "dark" ? "Dark" : "Light",
      })),
      dynamicTitle: true,
    },
  },
}

/**
 * The toolbar's choice, applied to the context the dashboard itself uses.
 *
 * Only when `theme` changes, so a story that owns a theme control
 * (`LoginPageShell`'s appearance button) wins over the toolbar rather than
 * being overwritten on its next render. The cost is that re-picking the value
 * the toolbar already holds does nothing: Storybook raises no update for it, so
 * after the in-page control has moved the preference, the toolbar restores it
 * on the other entry rather than the same one. Accepted, for a dev catalog.
 */
function StoryTheme({
  theme,
  children,
}: {
  theme: "light" | "dark"
  children: ReactNode
}) {
  const { setPreference } = useTheme()
  useEffect(() => {
    setPreference(theme)
  }, [theme, setPreference])
  return children
}

/** The catalog toolbar, driving the same theme context as the dashboard. */
export const withTheme: Decorator = (Story, context) => (
  <ThemeProvider>
    <StoryTheme theme={context.globals.theme === "dark" ? "dark" : "light"}>
      <Story />
    </StoryTheme>
  </ThemeProvider>
)
