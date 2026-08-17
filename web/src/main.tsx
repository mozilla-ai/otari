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
function loadBootstrap(): Promise<DeploymentBootstrap | null> {
  return apiFetch<DeploymentBootstrap>("/v1/bootstrap").catch(() => null)
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
