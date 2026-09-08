import { Component, type ReactNode } from "react"

import { PageError } from "./PageError"

/**
 * The catch above the router, and the only one the pre-session pages have.
 *
 * TanStack Router wraps its whole match tree in a `CatchBoundary` of its own
 * (`Matches.tsx`, unless `disableGlobalCatchBoundary`), so a throw inside the
 * shell already renders `router.tsx`'s `defaultErrorComponent`. `App`'s four
 * branches above `RouterProvider` (the sign-in screen, the public auth pages,
 * the invitation page, and the hybrid landing page) sit outside that, and a
 * throw in any of them unmounts to a blank document with nothing in it but a
 * console line. That is the worst place in the app to lose, because the sign-in
 * screen is the only page an operator can reach before every other one
 * (otari#806).
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

// `throw null` is legal, and a boundary that keys on the thrown value's own
// truthiness renders the child that threw it a second time, which React answers
// by unmounting to the root: the failure this file exists to prevent. So the
// flag is what decides, and this stands in where there is nothing to show.
const UNTYPED_FAILURE = new Error("Something went wrong.")

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
      <PageError error={this.state.error ?? UNTYPED_FAILURE}>
        The dashboard could not finish rendering this page. Reload to try again.
        If it keeps happening, this gateway and the dashboard it serves may be
        out of step, and restarting the gateway on a matching build is what puts
        them back.
      </PageError>
    )
  }
}
