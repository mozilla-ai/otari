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

// Which boundary catches a routed throw, kept here because this is what breaks
// when a library upgrade rewires it, and because reading the config gets it
// backwards. @tanstack/react-router has two: the global one (`Matches.tsx`)
// renders its CatchBoundary with no `errorComponent` and so always falls back to
// the built-in, while the per-match one (`Match.tsx`) wraps a match only when
// `errorComponent ?? defaultErrorComponent` is set, and sits inside the global
// one, so it catches first. Setting `defaultErrorComponent` is therefore what
// turns a usable boundary on rather than what styles an existing one. Asserted
// as a rendered outcome for the same reason.
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
