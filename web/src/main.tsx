import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import App from "@/app/App"
import { Provider } from "@/app/provider"
import type { DeploymentBootstrap } from "@/client"
import { apiFetch } from "@/shared/api/client"
import "@/styles/globals.css"

const container = document.getElementById("root")
if (!container) {
  throw new Error("Root element #root not found")
}

// Read before the first render rather than during it. The answer decides
// whether this URL is a management dashboard, a sign-in screen, or a data-plane
// gateway's landing page, and painting one of those only to swap it for another
// is worse than waiting one same-origin round trip. A failure hands App null,
// which says the gateway is unreachable instead of guessing a deployment.
// Shorter than apiFetch's 30s default, because nothing at all is on screen until
// this settles: a stalled gateway or proxy would otherwise hold a blank page for
// half a minute before the error state appears. A refused connection fails fast
// and never reaches this deadline; a healthy gateway answers immediately, since
// the route reads configuration plus one `LIMIT 1` probe for whether any
// identity holds a password (which credential the sign-in screen should ask
// for). A database the gateway cannot reach answers "no sign-in methods"
// instead of hanging, but a merely slow or pool-starved one does not: with
// `db_pool_timeout` defaulting to 30s, waiting for a connection can outlast
// this deadline, and then it is this timeout rather than the gateway that
// decides. Both land on the same screen, which says the gateway is unreachable.
const BOOTSTRAP_TIMEOUT_MS = 8_000

function loadBootstrap(): Promise<DeploymentBootstrap | null> {
  return apiFetch<DeploymentBootstrap>("/v1/bootstrap", {
    signal: AbortSignal.timeout(BOOTSTRAP_TIMEOUT_MS),
  }).catch(() => null)
}

void loadBootstrap().then((bootstrap) => {
  createRoot(container).render(
    <StrictMode>
      <Provider>
        <App bootstrap={bootstrap} />
      </Provider>
    </StrictMode>,
  )
})
