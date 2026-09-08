import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { router } from "@/app/router"

// Which boundary catches a routed throw is not obvious from the config, and it
// decides whether the failure wears the design system or the library's own
// inline-styled red `<pre>`. Two paths exist in @tanstack/react-router: the
// global one (`Matches.tsx`) renders its CatchBoundary with no errorComponent
// and therefore always falls back to the built-in, while the per-match one
// (`Match.tsx`) wraps each match only when `errorComponent ?? defaultErrorComponent`
// is set, and sits inside the global one so it catches first. Setting
// `defaultErrorComponent` is what turns the inner boundary on, and this asserts
// the outcome rather than the config, so a library upgrade that rewires it fails
// here instead of silently returning the red `<pre>`.
const rootRoute = createRootRoute()
const throwingRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: () => {
    throw new Error("routed page blew up")
  },
})

describe("router error handling", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders a routed throw on our own panel", async () => {
    // React logs a caught render error whatever catches it, and the router
    // warns about an uncaught one in development.
    vi.spyOn(console, "error").mockImplementation(() => {})
    vi.spyOn(console, "warn").mockImplementation(() => {})

    const probe = createRouter({
      routeTree: rootRoute.addChildren([throwingRoute]),
      history: createMemoryHistory({ initialEntries: ["/"] }),
      // The real one, so this tests the app's configuration and not a copy of it.
      defaultErrorComponent: router.options.defaultErrorComponent,
    })
    render(<RouterProvider router={probe} />)

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "routed page blew up",
    )
    // The built-in component's heading, which is what renders when the inner
    // boundary is off and the global one answers instead.
    expect(screen.queryByText("Something went wrong!")).toBeNull()
  })
})
