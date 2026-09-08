import { useEffect } from "react"
import type { Decorator } from "@storybook/react-vite"

import { THEME_PREFERENCES } from "@/shared/hooks/useTheme"

/**
 * A light/dark toolbar for the catalog.
 *
 * The dashboard's theme is not a React context that a decorator can just wrap:
 * it is three writes on the document element, and every consumer keys off a
 * different one of them. `globals.css` declares its dark token block under
 * `.dark, [data-theme="dark"]`, its `dark:` variant matches either spelling, and
 * the browser paints scrollbars and native controls from `color-scheme` alone.
 * Set only one and the page half-changes.
 *
 * So this is the third copy of that triple, after `ThemeProvider`
 * (`src/shared/hooks/useTheme.tsx`) and the pre-paint script in `index.html`,
 * which duplicates it because it has to run before React exists. Wrapping the
 * real `ThemeProvider` instead would not work: it reads the operator's stored
 * preference and owns its own state, so a toolbar could not drive it. Keep these
 * three writes in step with that effect.
 *
 * The preference names come from `THEME_PREFERENCES` rather than being restated,
 * minus "system": a catalog exists to show both themes deliberately, and
 * "whatever this laptop is set to" is not a case worth a toolbar entry.
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

export const withTheme: Decorator = (Story, context) => {
  const resolved = context.globals.theme === "dark" ? "dark" : "light"

  // In an effect rather than inline: the target is outside this React tree (the
  // iframe's own <html>), so writing it during render would be a side effect on
  // a node React does not own, and the React Compiler is free to re-run a render.
  useEffect(() => {
    const root = document.documentElement
    root.setAttribute("data-theme", resolved)
    root.classList.toggle("dark", resolved === "dark")
    root.style.colorScheme = resolved
  }, [resolved])

  return <Story />
}
