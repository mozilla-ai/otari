import type { Decorator, Preview } from "@storybook/react-vite"

import { withRouter } from "@/tests/router"
import "./preview.css"

import { withApiMocks } from "./apiMock"
import { withAppContext } from "./appContext"
import { themeGlobalType, withTheme } from "./theme"

/**
 * `preview.css` is `globals.css` plus the `@source` globs Tailwind needs to see
 * the story files at all (see the comment in there -- gitignored files are
 * invisible to its content detection). Through it comes the whole design system: the two token
 * blocks, the HeroUI alias layer that themes a bare `<Card>`, the `@font-face`
 * set, the type-scale utilities, and the `.otari-*` rules. A story never styles
 * from anything else, and `src/styles/foundation.test.ts` sweeps every story file
 * to make sure of it (raw hex, numbered Tailwind palette classes and
 * `bg-white`/`text-black` all fail there, over the whole of `src/`).
 */

/**
 * A live router around every story.
 *
 * `StatCard`'s `to`, `PublicAuthLink` and `Breadcrumbs` all render real links, and
 * a TanStack `Link` outside a router throws rather than degrading. This reuses the
 * test harness (`src/tests/router.tsx`) instead of building a second one: it
 * already carries the app's real `parseSearch`/`stringifySearch` codec and a splat
 * route under every unlisted path, so a link to anywhere resolves. Two router
 * harnesses would be two things to keep in step.
 *
 * `.storybook/` may import `@/tests` because it sits outside `src/`, so none of
 * biome.jsonc's layer overrides apply to it -- the same reason `src/tests/` is
 * allowed to reach `@/app`. A story file gets no such exemption, which is why
 * every provider is global and declared here rather than imported per story.
 *
 * Built once per story, since `withRouter()` returns a fresh router each call and
 * a component that navigates should not have its location reset by a re-render.
 */
const withAppRouter: Decorator = (Story) => {
  const Wrapper = withRouter()
  return (
    <Wrapper>
      <Story />
    </Wrapper>
  )
}

const preview: Preview = {
  globalTypes: themeGlobalType,

  // Innermost FIRST. Storybook composes with
  // `decorators.reduce((story, d) => d(story), storyFn)`, so each entry wraps the
  // ones before it and the *last* one ends up outermost. Written in the order the
  // providers nest, which is the reverse of how it reads:
  //
  //   withTheme( withAppRouter( withApiMocks( withAppContext( story ) ) ) )
  //
  // The nesting is not arbitrary. The theme writes to <html>, so it is outside
  // everything. `withAppContext` mounts `SelectedWorkspaceProvider`, which seeds
  // itself from a query, so it has to be *inside* the query client `withApiMocks`
  // provides -- getting this backwards fails every story that touches it with
  // "No QueryClient set".
  decorators: [withAppContext, withApiMocks, withAppRouter, withTheme],

  parameters: {
    layout: "centered",
    controls: { expanded: true },
    // Storybook paints its own canvas behind the story, which would sit a
    // hardcoded white under a dark-themed component. Off, so the token background
    // that `globals.css` gives <body> is what shows through.
    backgrounds: { disable: true },
  },
}

export default preview
