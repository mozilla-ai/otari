import { Spinner } from "@heroui/react"
import { useNavigate } from "@tanstack/react-router"
import { useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { ProductMark } from "@/design-system/ProductMark"
import { usePlatformConnectFlow } from "@/features/providers/PlatformConnect"
import {
  OWN_KEYS,
  readWelcomeAnswer,
  writeWelcomeAnswer,
} from "@/features/welcome/welcomeAnswer"
import { useHostedProviders } from "@/shared/api/hostedProviders"
import {
  useDeploymentOperator,
  useProviderKeyEncryption,
} from "@/shared/api/organizations"
import { useProviders } from "@/shared/api/providers"
import { useDeployment } from "@/shared/hooks/useDeployment"

const OFFERS = [
  "Open-weight models, hosted by Mozilla.ai",
  "Web search for your models",
  "Code execution in a hosted sandbox",
]

// As otari.ai's subscriptions page lists them. A sample, not the catalog.
const MODELS = ["glm-5.3", "deepseek-v4-pro", "kimi-k3", "qwen3.8-27b"]

const readAnswered = () => readWelcomeAnswer() !== null

/**
 * Whether to greet this operator, decided once when the reads settle, so the
 * connection this dialog makes does not close it before it can hand over.
 */
function useShouldWelcome(): boolean {
  // The answer itself rather than `isOperator`, which fails open: an outage
  // is no reason to greet a tenant.
  const { answer, isSettled } = useDeploymentOperator()
  const canStoreKey = useProviderKeyEncryption()
  const enabled = answer === "operator"
  const configured = useProviders(enabled)
  const hosted = useHostedProviders(enabled)
  const [decision, setDecision] = useState<"pending" | "show" | "hide">(
    "pending",
  )
  if (
    decision === "pending" &&
    isSettled &&
    (!enabled || (!hosted.isPending && !configured.isPending))
  ) {
    // Any provider, configured or hosted, already answers where models come
    // from; otari.ai being one of them is the connected case.
    const hasProvider =
      (configured.data?.providers ?? []).length > 0 ||
      (hosted.data?.data ?? []).length > 0
    setDecision(
      enabled &&
        canStoreKey &&
        !hosted.isError &&
        !configured.isError &&
        !hasProvider
        ? "show"
        : "hide",
    )
  }
  return decision === "show"
}

/**
 * The first thing a self-hosted deployment's operator sees: a welcome that
 * offers otari.ai's models, once, and is gone for good however it is
 * closed.
 *
 * Only an operator who could connect is greeted, and never on a gateway
 * that already serves a provider, otari.ai or another. Nothing is sent to otari.ai unless Connect
 * is pressed. Mounted from the base `PostSignInGate`, which otari.ai's own
 * build replaces, so it never renders there.
 */
export function WelcomeDialog() {
  const { platform_connect } = useDeployment()
  // Read before anything is fetched, so an operator who answered asks nothing.
  const [answeredAtLoad] = useState(readAnswered)
  return platform_connect && !answeredAtLoad ? <Welcome /> : null
}

function Welcome() {
  const shouldWelcome = useShouldWelcome()
  const [answered, setAnswered] = useState(false)
  if (!shouldWelcome || answered) {
    return null
  }
  const answer = (value: string) => {
    writeWelcomeAnswer(value)
    setAnswered(true)
  }
  return (
    <WelcomeFrame
      onClose={() => answer(new Date().toISOString())}
      onSkip={() => answer(OWN_KEYS)}
    />
  )
}

function WelcomeFrame({
  onClose: close,
  onSkip,
}: {
  onClose: () => void
  onSkip: () => void
}) {
  const navigate = useNavigate()
  const flow = usePlatformConnectFlow(() => {
    close()
    // The Overview's setup guide hands out a key and the first request.
    void navigate({ to: "/" })
  })
  const isPending = flow.state === "pending"
  // Answering ends a code still waiting, or the providers page would offer it
  // again straight after. Not on the approval, which closes through `close`.
  const endWaiting = () => {
    if (isPending) flow.cancel()
  }

  return (
    <Dialog
      isOpen
      onOpenChange={(open) => {
        if (!open) {
          endWaiting()
          close()
        }
      }}
      title="Welcome to Otari"
      description="Your gateway is running. Point any app at it and Otari routes, meters and guards every request. Bring your own provider keys, or use ours."
      mark={<ProductMark className="text-accent mt-1 h-auto w-9 shrink-0" />}
      isAnnouncement
      isScanning={isPending}
      status={
        isPending ? (
          <div className="flex flex-col gap-1" role="status">
            <span className="flex items-center gap-2 text-sm text-muted">
              <Spinner size="sm" aria-hidden="true" />
              Waiting for you to approve this code on otari.ai
            </span>
            <span className="text-mono-caption text-lg text-foreground">
              {flow.userCode}
            </span>
            {flow.link ? (
              <a
                href={flow.link}
                target="_blank"
                rel="noreferrer"
                className="text-sm font-medium text-link hover:text-link-hover"
              >
                Open the approval page
              </a>
            ) : null}
          </div>
        ) : flow.ended ? (
          <span className="text-sm" role="status">
            {flow.ended}
          </span>
        ) : undefined
      }
      footerStart={
        <button
          type="button"
          className="min-h-11 text-sm text-muted hover:text-foreground md:min-h-9"
          onClick={() => {
            endWaiting()
            onSkip()
          }}
        >
          Skip, I&rsquo;ll use my own keys
        </button>
      }
      actions={
        isPending ? (
          <Button
            className="min-h-11 md:min-h-9"
            isPending={flow.isCancelling}
            onPress={flow.cancel}
          >
            Cancel
          </Button>
        ) : (
          <Button
            variant="primary"
            className="min-h-11 md:min-h-9"
            isPending={flow.isStarting}
            onPress={flow.connect}
          >
            {flow.ended ? "Try again" : "Connect otari.ai"}
          </Button>
        )
      }
    >
      <DialogSection>
        <div className="border-accent bg-primary-subtle/40 flex flex-col gap-3 border p-4">
          <div className="flex items-center gap-2 font-medium text-foreground">
            <span className="bg-foreground text-background flex size-6 items-center justify-center">
              <ProductMark className="h-auto w-4" />
            </span>
            Use otari.ai with this gateway
          </div>
          <ul className="text-body flex flex-col gap-2">
            {OFFERS.map((offer) => (
              <li key={offer} className="flex items-baseline gap-3">
                <span
                  aria-hidden
                  className="bg-accent size-1.5 shrink-0 -translate-y-0.5"
                />
                <span>{offer}</span>
              </li>
            ))}
          </ul>
          <ul aria-label="Some of its models" className="flex flex-wrap gap-2">
            {MODELS.map((model) => (
              <li
                key={model}
                className="text-mono-caption border-border bg-surface border px-2 py-0.5"
              >
                {model}
              </li>
            ))}
          </ul>
          <p className="text-sm text-muted">
            Pay as you go with credits, or take a subscription. Requests you
            send to otari.ai leave this gateway.
          </p>
        </div>
        <ErrorBanner error={flow.error} />
      </DialogSection>
    </Dialog>
  )
}
