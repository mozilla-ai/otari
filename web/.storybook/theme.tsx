import { type ReactNode, useEffect } from "react"
import type { Decorator } from "@storybook/react-vite"

import { THEME_PREFERENCES, ThemeProvider, useTheme } from "@/shared/hooks/useTheme"

/**
 * The catalog toolbar drives the same theme context as the dashboard.
 *
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

export const withTheme: Decorator = (Story, context) => (
  <ThemeProvider>
    <StoryTheme theme={context.globals.theme === "dark" ? "dark" : "light"}>
      <Story />
    </StoryTheme>
  </ThemeProvider>
)
