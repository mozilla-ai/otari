import { useEffect, useRef, useState, useSyncExternalStore } from "react"
import App from "@/app/App"
import { announce } from "@/app/announce"
import {
  BOOT_FAILURE_COPY,
  BootFailureView,
  bootFailureTitle,
} from "@/app/BootFailureView"
import { BootConnecting } from "@/app/BootFrame"
import {
  type BootLoader,
  describeBootFailure,
  type FailedState,
} from "@/app/boot"
import { requestPolicy } from "@/shared/api/overlayRequestPolicy"

/**
 * The host the bootstrap was asked of: the page's own, unless the request
 * policy points the API at another origin, in which case that one, since it
 * is the address that did not answer.
 */
function askedHost(): string {
  const { origin } = requestPolicy()
  return origin === "" ? window.location.host : new URL(origin).host
}

/** Read aloud when a pressed Retry fails the way the last attempt did. */
const STILL_NO_ANSWER = "Still no answer."

/**
 * The app once the bootstrap has answered, and until then the reason it has
 * not. There is deliberately no deployment to fall back to: rendering a
 * management dashboard at a gateway that does not serve one is the failure the
 * bootstrap contract exists to prevent, so a failed read says so and keeps
 * asking, and the app mounts the moment an attempt succeeds.
 */
export function DeploymentGate({ loader }: { loader: BootLoader }) {
  const state = useSyncExternalStore(loader.subscribe, loader.getState)

  if (state.status === "ready") {
    return <App bootstrap={state.bootstrap} />
  }
  // `main.tsx` mounts after the first attempt settles, so this is only
  // reachable from a caller that mounts earlier. The same words `index.html`
  // paints while the bundle loads.
  if (state.status === "loading") {
    return <BootConnecting />
  }

  return <BootFailure state={state} loader={loader} />
}

/**
 * The failure screen's behavior: which press is pending, what a screen reader
 * is told, and the page title.
 *
 * Pending belongs to a press, not to an attempt. Background attempts run every
 * few seconds and a button that went pending for each one spent most of a long
 * outage refusing to be pressed. A press that lands while a background attempt
 * is already running adopts it (`retryNow` hands back the running attempt), so
 * it is answered by that attempt rather than queued behind it.
 *
 * A screen reader hears three things and nothing else: the first failure, a
 * change in which failure it is, and a pressed Retry failing. A change is a
 * change of heading: the four retrying failures share theirs, so a gateway
 * flapping between no response and a 503 is one failure to the listener, and
 * the Last try row is where the difference shows. Focus is left where it is.
 */
function BootFailure({
  state,
  loader,
}: {
  state: FailedState
  loader: BootLoader
}) {
  const failure = describeBootFailure(state)
  const [isPressed, setIsPressed] = useState(false)
  // The failure a pressed Retry ended in, so the row says so until the next
  // attempt replaces it.
  const [pressFailedAt, setPressFailedAt] = useState<number | null>(null)
  const heading = BOOT_FAILURE_COPY[failure.kind].heading
  const announced = useRef<string | null>(null)

  useEffect(() => {
    if (announced.current !== heading) {
      announced.current = heading
      announce(heading)
    }
  }, [heading])

  useEffect(() => {
    const previous = document.title
    document.title = bootFailureTitle(failure.kind)
    return () => {
      document.title = previous
    }
  }, [failure.kind])

  function onAction() {
    if (BOOT_FAILURE_COPY[failure.kind].action === "reload") {
      window.location.reload()
      return
    }
    const pressedOn = heading
    setIsPressed(true)
    void loader.retryNow().then(() => {
      setIsPressed(false)
      const settled = loader.getState()
      if (settled.status !== "failed") {
        return
      }
      setPressFailedAt(settled.failedAt)
      const copy = BOOT_FAILURE_COPY[describeBootFailure(settled).kind]
      // A different failure is announced by its heading, above.
      if (copy.heading === pressedOn) {
        announce(copy.action === "retryNow" ? STILL_NO_ANSWER : copy.heading)
      }
    })
  }

  return (
    <BootFailureView
      failure={failure}
      failedAt={state.failedAt}
      host={askedHost()}
      isPending={isPressed}
      isStillNoAnswer={
        pressFailedAt === state.failedAt &&
        BOOT_FAILURE_COPY[failure.kind].action === "retryNow"
      }
      onAction={onAction}
    />
  )
}
