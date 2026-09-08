import { Component, type ReactNode } from "react"

import { ErrorBanner } from "./ErrorBanner"

/**
 * The catch above the router, and the only one the pre-session pages have.
 *
 * TanStack Router wraps every routed page in a `CatchBoundary` of its own, so a
 * throw inside the shell already renders its "Something went wrong!" panel.
 * `App`'s four branches above `RouterProvider` (the sign-in screen, the public
 * auth pages, the invitation page, and the hybrid landing page) sit outside
 * that, and a throw in any of them unmounts to a blank document with nothing in
 * it but a console line. That is the worst place in the app to lose, because
 * the sign-in screen is the only page an operator can reach before every other
 * one (otari#806).
 *
 * A class because React offers no hook for this; `getDerivedStateFromError` and
 * `componentDidCatch` are still the only way to catch a render.
 *
 * Nothing is logged here on purpose: `src/` writes to no console anywhere, and
 * the error is on screen, which is the whole point of the panel.
 */
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: unknown }
> {
  state: { error: unknown } = { error: null }

  static getDerivedStateFromError(error: unknown): { error: unknown } {
    return { error }
  }

  render() {
    if (!this.state.error) {
      return this.props.children
    }
    // The same shape as `App`'s unreachable-gateway panel, because they are the
    // same situation to the person reading one: the dashboard is up and cannot
    // show them the deployment.
    return (
      <div className="flex min-h-full items-center justify-center p-6">
        <div className="flex w-full max-w-md flex-col gap-3">
          <ErrorBanner error={this.state.error} />
          <p className="text-caption">
            The dashboard could not finish rendering this page. Reload to try
            again. If it keeps happening, this gateway and the dashboard it
            serves may be out of step, and restarting the gateway on a matching
            build is what puts them back.
          </p>
        </div>
      </div>
    )
  }
}
