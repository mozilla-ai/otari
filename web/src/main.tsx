// First: it moves an old `/#/…` link to its path before the router reads the address.
import "@/app/legacyHashRouteOnLoad"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import { reloadOnStaleChunk } from "@/app/AutoUpdate"
import { createBootLoader, firstSettled, readBootstrap } from "@/app/boot"
import { DeploymentGate } from "@/app/DeploymentGate"
import { Provider } from "@/app/provider"
import { captureSetupKey } from "@/features/auth/setupKey"
import { prepareRequests } from "@/shared/api/overlayRequestPolicy"
import { currentRoute } from "@/shared/helpers/appLocation"
import "@/styles/globals.css"

reloadOnStaleChunk()

// Before anything is fetched or rendered, so a first-run setup link's key
// leaves the address bar at once rather than after the bootstrap answers.
captureSetupKey(currentRoute())

const container = document.getElementById("root")
if (!container) {
  throw new Error("Root element #root not found")
}

// Mounted once the first bootstrap attempt settles rather than at once. The
// answer decides whether this URL is a management dashboard or a sign-in
// screen, and painting one only to swap it for the other is worse than waiting
// one same-origin round trip. A failed first attempt mounts the failure screen,
// which keeps retrying and becomes the app when an attempt answers
// (`app/boot.ts`). The request policy is settled before the bootstrap is asked
// for, because the answer can change where that request goes.
const loader = createBootLoader({
  prepare: prepareRequests,
  load: readBootstrap,
})

void firstSettled(loader).then(() => {
  createRoot(container).render(
    <StrictMode>
      <Provider>
        <DeploymentGate loader={loader} />
      </Provider>
    </StrictMode>,
  )
})
