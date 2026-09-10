import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Button } from "../actions/Button"
import { ErrorBoundary } from "./ErrorBoundary"

/**
 * The catch above the router, and the only one the pre-session pages have.
 *
 * Everything inside `RouterProvider` is already caught, so this covers `App`'s
 * four branches above it: the sign-in screen, the public auth pages, the
 * invitation page, and the hybrid landing. A throw in one of those unmounts to
 * a blank document, which is the worst place in the app to lose, because the
 * sign-in screen is the only page reachable before every other one.
 *
 * A class component because React still offers no hook for this:
 * `getDerivedStateFromError` and `componentDidCatch` are the only way to catch
 * a render.
 *
 * **Nothing throws until you press.** That is deliberate rather than a weaker
 * demonstration: React writes a caught error to `console.error`, and the
 * catalog's smoke run treats a console error on load as a failing story. Behind
 * a press, the story loads clean and still shows the real thing.
 */
// `component` is deliberately absent. Storybook's inference cannot type a class
// component whose `getDerivedStateFromProps` is declared over its own props, and
// every story here supplies a `render` anyway, since a boundary needs a child
// that throws to be worth looking at.
const meta = {
  title: "Design system/Feedback/ErrorBoundary",
  parameters: { layout: "fullscreen" },
} satisfies Meta

export default meta

type Story = StoryObj

function Boom(): never {
  throw new Error("Cannot read properties of undefined (reading 'map')")
}

/** Press to throw inside the boundary and see the panel it renders. */
export const Default: Story = {
  render: () => {
    const [broken, setBroken] = useState(false)
    return (
      <ErrorBoundary>
        <div className="p-6">
          {broken ? (
            <Boom />
          ) : (
            <Button variant="danger" onPress={() => setBroken(true)}>
              Throw inside the boundary
            </Button>
          )}
        </div>
      </ErrorBoundary>
    )
  },
}

/**
 * `resetKey` clears the caught state when it changes, the way the router's own
 * boundary takes `getResetKey`. Without one the panel latches: `App` picks its
 * branch from the hash, so following a second emailed link after the first one
 * threw would keep showing the first link's failure.
 *
 * Throw, then change the key, and the child renders again.
 */
export const ResetKey: Story = {
  render: () => {
    const [broken, setBroken] = useState(false)
    const [key, setKey] = useState("first-link")
    return (
      <div className="flex flex-col gap-3 p-6">
        <div className="flex items-center gap-3">
          <Button variant="danger" onPress={() => setBroken(true)}>
            Throw
          </Button>
          <Button
            onPress={() => {
              setBroken(false)
              setKey(key === "first-link" ? "second-link" : "first-link")
            }}
          >
            Change resetKey
          </Button>
          <span className="font-mono text-mono-caption">resetKey: {key}</span>
        </div>
        <ErrorBoundary resetKey={key}>
          {broken ? <Boom /> : <p className="text-body">Rendering normally.</p>}
        </ErrorBoundary>
      </div>
    )
  },
}
