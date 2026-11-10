import { useEffect, useRef } from "react"

import type { PlatformConnectStatus } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import {
  useCancelPlatformConnect,
  usePlatformConnectStatus,
  useStartPlatformConnect,
} from "@/shared/api/hostedProviders"

/** Where an operator revokes the key a connection issued. */
export const PLATFORM_KEYS_URL = "https://otari.ai/dashboard/#/keys"

/** The hosted provider a connection stores. */
export const PLATFORM_PROVIDER_NAME = "otari-ai"

const ENDED_COPY: Partial<Record<PlatformConnectStatus["status"], string>> = {
  expired: "The code expired before it was approved.",
  denied: "The request was declined on otari.ai.",
}

/** The approval page as a link target, or empty when it is not a web address. */
function approvalLink(status: PlatformConnectStatus | undefined): string {
  const link = status?.verification_uri_complete ?? ""
  return /^https?:\/\//i.test(link) ? link : ""
}

/**
 * Connect this gateway to otari.ai, in place: the offer, then the code to
 * approve with its link and a waiting state, then how it ended.
 *
 * The gateway asks otari.ai and stores the key; this only starts the attempt,
 * opens the approval page and follows the status. Nothing is sent to otari.ai
 * before Connect is pressed. Render it only where the bootstrap offers
 * `platform_connect`.
 */
export function PlatformConnect({
  onConnected,
}: {
  /** Called once, when an attempt this component watched is approved. */
  onConnected?: () => void
}) {
  const status = usePlatformConnectStatus(true)
  const start = useStartPlatformConnect()
  const cancel = useCancelPlatformConnect()
  const state = status.data?.status ?? "idle"

  // Only an approval this component waited for: a deployment connected earlier
  // must not send whoever opens the page somewhere else.
  const wasPending = useRef(false)
  useEffect(() => {
    if (state === "pending") {
      wasPending.current = true
    } else if (state === "connected" && wasPending.current) {
      wasPending.current = false
      onConnected?.()
    }
  }, [state, onConnected])

  const connect = () => {
    // Opened while the press is still the active gesture: a tab opened after
    // the gateway answers is one a browser blocks. The link stays on the page
    // for a browser that blocks this one too.
    const tab = window.open("", "_blank")
    start.mutate(undefined, {
      onSuccess: (started) => {
        const link = approvalLink({ status: "pending", ...started })
        if (!tab) return
        if (!link) {
          tab.close()
          return
        }
        tab.opener = null
        tab.location.replace(link)
      },
      onError: () => tab?.close(),
    })
  }

  const link = approvalLink(status.data)
  const ended = ENDED_COPY[state]

  return (
    <div className="flex flex-col gap-3">
      <div>
        <h3 className="text-title">Connect otari.ai</h3>
        <p className="mt-1 text-sm text-muted">
          One key for models from many providers. otari.ai is a paid service,
          and requests sent to it leave this gateway.
        </p>
      </div>

      {state === "pending" ? (
        <div className="flex flex-col gap-3" role="status">
          <p className="text-sm text-muted">
            Approve this code on otari.ai. This page updates on its own.
          </p>
          <p className="text-mono-caption text-lg text-foreground">
            {status.data?.user_code}
          </p>
          <div className="flex flex-wrap items-center gap-3">
            {link ? (
              <a
                href={link}
                target="_blank"
                rel="noreferrer"
                className="text-sm font-medium text-link hover:text-link-hover"
              >
                Open the approval page
              </a>
            ) : null}
            <Button
              className="min-h-11 md:min-h-9"
              isPending={cancel.isPending}
              onPress={() => cancel.mutate()}
            >
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {state === "connected" ? (
            <p className="text-sm text-foreground" role="status">
              Connected. Models are served as{" "}
              <code>otari-ai:&lt;vendor&gt;:&lt;model&gt;</code>.
            </p>
          ) : null}
          {ended ? (
            <p className="text-sm text-foreground" role="status">
              {ended}
            </p>
          ) : null}
          <div>
            <Button
              variant="primary"
              className="min-h-11 md:min-h-9"
              isPending={start.isPending}
              onPress={connect}
            >
              {ended
                ? "Try again"
                : state === "connected"
                  ? "Connect again"
                  : "Connect otari.ai"}
            </Button>
          </div>
        </div>
      )}

      <ErrorBanner error={start.error ?? cancel.error ?? status.error} />
    </div>
  )
}
