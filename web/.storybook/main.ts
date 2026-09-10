import type { StorybookConfig } from "@storybook/react-vite"
import type { PluginOption } from "vite"

/**
 * A component catalog for the dashboard's primitives.
 *
 * This directory is absent from `biome.jsonc`'s `files.includes` and from every
 * `tsconfig` leaf, so the config here is neither linted nor typechecked; adding it
 * to both is open work. The stories themselves live under `src/`, so they *are*
 * linted, typechecked, and swept by `src/styles/foundation.test.ts` like any other
 * source.
 *
 * Config, not stories, lives here. Stories are colocated beside the component
 * they document, the way `Foo.test.tsx` sits beside `Foo.tsx`, and no directory
 * is added under `src/`, because `src/architecture.test.ts` asserts the layer set
 * there is closed (app, client, features, routes, shared, styles, tests).
 */

/**
 * Every plugin whose name starts with `tanstack`, flattened out of the app's
 * config.
 *
 * `tanstack:router-generator` rewrites `src/routeTree.gen.ts`, which is committed
 * and guarded in CI by `git diff --exit-code`; the `tanstack-router:code-splitter`
 * ones rewrite the route modules. Both would produce identical content today, but
 * this catalog must not be able to dirty a tracked generated file, and the
 * catalog renders components rather than routes, so none of them have anything to
 * do here. Stories that need a router get one from the decorator in `preview.tsx`.
 *
 * The flatten is the load-bearing part: `tanstackRouter()` returns a `Plugin[]`,
 * not a `Plugin` (so does `react()`), so a filter over the top level alone sees an
 * unnamed array and keeps the whole thing. Vite accepts a flat array, so
 * flattening the rest is free.
 */
function withoutTanstackPlugins(plugins: PluginOption[]): PluginOption[] {
  return plugins.flat(Infinity as 1).filter((plugin) => {
    const name =
      plugin && typeof plugin === "object" && "name" in plugin
        ? String(plugin.name)
        : ""
    return !name.startsWith("tanstack")
  })
}

const config: StorybookConfig = {
  // The catalog, plus the measuring harness when it is asked for.
  //
  // A harness renders a component in every context some stylesheet rule targets
  // so a script can read the computed geometry before and after a change. It is
  // not a catalog entry, and it used to try to say so with a `!` pattern in this
  // array. **Storybook does not support negation here**: the entry was ignored
  // and `Zz-measure/TableGeometry` published to the live site, in the sidebar,
  // for anyone reading the design system to trip over.
  //
  // So the harness sits outside `../src/**` instead, where the default glob
  // cannot reach it, and is opted into by an environment variable. That also
  // keeps it out of the smoke run, which is what a harness wants: it renders one
  // enormous story whose only reader is a script.
  //
  //   STORYBOOK_HARNESS=1 pnpm --dir web run storybook
  stories: [
    "../src/**/*.stories.tsx",
    ...(process.env.STORYBOOK_HARNESS ? ["./harness/*.stories.tsx"] : []),
  ],
  addons: ["@storybook/addon-docs"],
  framework: { name: "@storybook/react-vite", options: {} },

  // The dashboard declares no analytics vendor (see web/AGENTS.md), so Storybook's
  // own usage reporting is off too.
  core: { disableTelemetry: true },

  // Storybook loads and merges web/vite.config.ts, which is what gets the catalog
  // the app's own pipeline: the `@` alias, Tailwind v4, and the React Compiler
  // Babel pass, so a story runs under the same memoization the app does.
  viteFinal: async (viteConfig) => {
    viteConfig.plugins = withoutTanstackPlugins(viteConfig.plugins ?? [])

    // The merged config brings the app's `base: "/"` with it, which is right for
    // a gateway serving the dashboard at an origin root and wrong for a catalog
    // published under a repository path: GitHub Pages serves this at
    // `/<repo>/`, and every asset URL the build emits would point one directory
    // too high. Read from the environment rather than hardcoded, so the same
    // build serves the dev server, a local `storybook build`, and Pages without
    // three configs. Same problem the app's own `base` had in #857.
    const base = process.env.STORYBOOK_BASE_PATH
    if (base) {
      viteConfig.base = base
    }
    return viteConfig
  },
}

export default config
