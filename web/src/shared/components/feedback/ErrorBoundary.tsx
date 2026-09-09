import { Component, type ReactNode } from "react"

import { PageError } from "./PageError"

/**
 * The catch above the router, and the only one the pre-session pages have.
 *
 * Everything inside `RouterProvider` is already caught, so this covers only
 * `App`'s four branches above it: the sign-in screen, the public auth pages,
 * the invitation page, and the hybrid landing page. A throw in one of those
 * unmounts to a blank document with nothing in it but a console line, which is
 * the worst place in the app to lose, because the sign-in screen is the only
 * page an operator can reach before every other one (otari#806).
 * `router.test.tsx` has which boundary answers where.
 *
 * A class because React offers no hook for this; `getDerivedStateFromError` and
 * `componentDidCatch` are still the only way to catch a render.
 *
 * Nothing is logged here on purpose: `src/` writes to no console anywhere, and
 * the error is on screen, which is the whole point of the panel.
 */

interface Props {
  children: ReactNode
  /**
   * Clears the caught state when it changes, the way the router's own boundary
   * takes `getResetKey`. Without one the panel latches: `App` picks the branch
   * to render from the hash, so following a second emailed link after one of
   * them threw would keep showing the first link's failure.
   */
  resetKey?: string
}

interface State {
  caught: boolean
  error: unknown
  /** The key the current caught state belongs to, to compare the next one against. */
  seenKey: string | undefined
}

// A falsy throw is legal, and a boundary that keys on the thrown value's own
// truthiness renders the child that threw it a second time, which React answers
// by unmounting to the root: the failure this file exists to prevent. So the
// flag is what decides. Showing such a value is `PageError`'s end.

export class ErrorBoundary extends Component<Props, State> {
  state: State = { caught: false, error: undefined, seenKey: undefined }

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { caught: true, error }
  }

  static getDerivedStateFromProps(props: Props, state: State): State | null {
    if (props.resetKey === state.seenKey) {
      return null
    }
    return { caught: false, error: undefined, seenKey: props.resetKey }
  }

  render() {
    if (!this.state.caught) {
      return this.props.children
    }
    return (
      <PageError error={this.state.error}>
        The dashboard could not finish rendering this page. Reload to try again.
        If it keeps happening, this gateway and the dashboard it serves may be
        out of step, and restarting the gateway on a matching build is what puts
        them back.
      </PageError>
    )
  }
}
