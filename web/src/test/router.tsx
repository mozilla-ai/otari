import {
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import { act, render } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";

import { parseSearch, stringifySearch, validateSearch } from "@/lib/search";

export interface TestRouterOptions {
  /** Where the router starts, query string included. */
  url?: string;
  /** Extra destinations, so a test can observe where a navigation landed. */
  routes?: { path: string; element: ReactNode }[];
  /**
   * A layout to render every route inside, for testing the shell itself. It
   * must contain an `<Outlet>`, which is where the routes below it appear.
   */
  shell?: ReactNode;
}

/**
 * A router around one component, standing in for the real route tree.
 *
 * The pages keep their filters, pagination and drill-downs in the URL, so they
 * need a live router rather than a stub. This mounts the component under test at
 * `url` and nothing else, apart from whatever destinations the test wants to
 * observe. It is returned as a `wrapper`, which is what `render` and
 * `renderHook` take.
 *
 * The router is built once per `withRouter()` call, so each test gets its own
 * history and a re-render never resets the location mid-test.
 *
 * Memory history, where the app runs on hash history, so a `Link` renders
 * `/activity?status=error` here and `/#/activity?status=error` in the browser.
 * That is deliberate: hash history in jsdom would share one global
 * `window.location` across tests and make them order-dependent. It does mean an
 * href assertion below pins the path and query but not the shipped hash prefix,
 * so the hash form is asserted in `web/e2e` instead.
 */
export function withRouter({ url = "/", routes = [], shell }: TestRouterOptions = {}) {
  const path = url.split("?")[0] || "/";
  // The wrapper's children are not known until it renders, so the route reads
  // them from here. Safe because the router hooks subscribe to router state
  // themselves: a navigation re-renders the component under test directly,
  // rather than relying on this indirection to propagate anything.
  const slot: { current: ReactNode } = { current: null };

  const rootRoute = createRootRoute({
    validateSearch,
    component: shell === undefined ? undefined : () => <>{shell}</>,
  });
  const subject = createRoute({ getParentRoute: () => rootRoute, path, component: () => <>{slot.current}</> });
  const probes = routes.map((route) =>
    createRoute({ getParentRoute: () => rootRoute, path: route.path, component: () => <>{route.element}</> }),
  );
  // A splat under every unlisted path, so a link or navigation the test does not
  // care about still resolves instead of failing on a missing route.
  const rest = createRoute({ getParentRoute: () => rootRoute, path: "$", component: () => null });

  const router = createRouter({
    routeTree: rootRoute.addChildren([subject, ...probes, rest]),
    history: createMemoryHistory({ initialEntries: [url] }),
    // The same search codec the app runs on, so a test URL is read (and
    // rewritten) exactly as the browser would read it.
    parseSearch,
    stringifySearch,
  });

  return function Wrapper({ children }: { children: ReactNode }) {
    slot.current = children;
    return <RouterProvider router={router} />;
  };
}

/**
 * Let the router mount its first match.
 *
 * It resolves the initial location from a layout effect, so the synchronous
 * render puts nothing in the DOM. A test that then queries with `findBy*` waits
 * this out on its own; one that reads the DOM (or a hook result) straight away
 * has to flush it first.
 */
export function flushRouter(): Promise<void> {
  return act(async () => {});
}

/** `render` into a test router, already mounted. See {@link flushRouter}. */
export async function renderWithRouter(ui: ReactElement, options: TestRouterOptions = {}) {
  const result = render(ui, { wrapper: withRouter(options) });
  await flushRouter();
  return result;
}
