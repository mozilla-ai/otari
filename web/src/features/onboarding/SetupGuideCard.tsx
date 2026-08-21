import { Button, Card } from "@heroui/react"
import { Link, useNavigate } from "@tanstack/react-router"
import { useRef, useState } from "react"

import type { ActivationApiKey, ActivationAttempt } from "@/client"
import { setupFailureCopy } from "@/features/onboarding/setupFailureCopy"
import {
  useCreateActivationKey,
  useDismissActivation,
  useModels,
  useWorkspaceActivation,
} from "@/shared/api/hooks"
import { CopyField, ErrorBanner, InfoBanner } from "@/shared/components/ui"
import { formatCost, formatRelative } from "@/shared/helpers/format"
import {
  buildCurlSnippet,
  buildPythonSnippet,
  SNIPPET_MODEL_PLACEHOLDER,
} from "@/shared/helpers/requestSnippets"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * The step between "a provider is configured" and "this dashboard has something
 * to show": a workspace's first successful request.
 *
 * It composes with the Overview's own getting-started panel rather than
 * competing with it. That one is shown while the gateway has no provider, when a
 * request cannot succeed yet; this one takes over once one exists, hands out an
 * API key scoped to the selected workspace, shows the two calls that use it, and
 * watches the workspace's traffic until one lands.
 *
 * **Deliberately a panel and not a modal.** The platform's equivalent
 * (`otari-ai` `frontend/src/features/onboarding`) is a blocking sheet over the
 * whole app, which suits a hosted signup: the account is new, the first request
 * is the only thing to do, and nothing else on the page matters yet. A
 * self-hosted operator arrives with work in hand (a provider to configure, a key
 * to mint for someone else), so the guide sits at the top of the page they land
 * on, offers itself, and gets out of the way.
 *
 * The key is minted on a press rather than on render, for the same reason: a
 * credential nobody asked for should not exist because somebody opened the
 * Overview page.
 */
export function SetupGuideCard({
  hasProviders,
}: {
  /**
   * Whether the deployment has a provider configured, which is the caller's
   * answer to give rather than this card's to fetch.
   *
   * Not a `useProviders()` call in here, and that is load-bearing: the page
   * above decides whether to render at all from the *fetching* state of that
   * same query, so a second observer inside a child re-triggers it on mount,
   * flips the page back to its loading branch, and unmounts the observer that
   * asked, which remounts and asks again. One query, one owner.
   */
  hasProviders: boolean
}) {
  const surfaces = useSurfaces()
  const { selected } = useSelectedWorkspace()
  const workspaceId = selected?.workspace_id ?? null
  // Held back until the deployment can actually serve a request: with no
  // provider the Overview's own getting-started panel is the right guide, and
  // this one would be handing out a key for a call that cannot succeed.
  const activation = useWorkspaceActivation(
    workspaceId,
    surfaces("workspaces") && hasProviders,
  )
  const createKey = useCreateActivationKey()
  const dismiss = useDismissActivation()
  const [issued, setIssued] = useState<ActivationApiKey>()
  const [finished, setFinished] = useState(false)
  // The card only celebrates a first request it was present for. Without this
  // latch, an operator who never opened the guide would be congratulated on the
  // traffic they already had, on the next page load after it arrived.
  const wasOffered = useRef(false)

  const data = activation.data
  if (data?.experience_eligible) {
    wasOffered.current = true
  }
  const activated = data?.status === "activated"
  const showing = Boolean(data) && (data?.experience_eligible || activated)

  // Nothing at all rather than an explanation: the guide is an offer, and a
  // workspace that is set up (or one whose check failed) is better served by the
  // page it came for.
  if (!workspaceId || !showing || finished) {
    return null
  }
  if (activated) {
    return wasOffered.current ? (
      <SetupComplete
        attempt={data?.activation_attempt ?? undefined}
        onDone={() => setFinished(true)}
      />
    ) : null
  }

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-6">
        <div className="flex flex-col gap-1">
          {/* `text-heading`, the section role from the type scale, rather
              than a size spelled inline: this card sits at the top of the page
              beside the Overview's own getting-started panel and reads as its
              peer. */}
          <h2 className="text-heading">Send your first request</h2>
          <p className="text-sm text-muted">
            It lands in{" "}
            <span className="font-medium text-foreground">
              {selected?.name ?? "this workspace"}
            </span>
            . Usage, spend and the activity log stay empty until one does, so
            this guide watches for it and finishes here.
          </p>
        </div>

        {issued ? (
          <IssuedKey issued={issued} />
        ) : (
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="primary"
              isPending={createKey.isPending}
              onPress={() =>
                createKey.mutate(workspaceId, {
                  onSuccess: (key) => setIssued(key),
                })
              }
            >
              Create a setup key
            </Button>
            <span className="text-xs text-muted">
              An API key for this workspace, shown once. It appears on the API
              keys page as “Setup guide”.
            </span>
          </div>
        )}
        <ErrorBanner error={createKey.error} />

        <ListeningRow
          attempt={
            data?.status === "failed"
              ? (data.latest_attempt ?? undefined)
              : undefined
          }
          isChecking={activation.isFetching}
          checkFailed={activation.isError}
          onCheckNow={() => void activation.refetch()}
        />

        <ErrorBanner error={dismiss.error} />
        <div className="flex flex-wrap items-center justify-end gap-3">
          <Button
            variant="ghost"
            isPending={dismiss.isPending}
            onPress={() => dismiss.mutate(workspaceId)}
          >
            Skip this guide
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

/** The key, and the two calls that use it, once the operator has asked for one. */
function IssuedKey({ issued }: { issued: ActivationApiKey }) {
  const models = useModels()
  // The first model the gateway can serve, so the snippets are runnable as
  // pasted. With none (a provider that answers no listing) the placeholder
  // stands and the copy below says what to do about it.
  const model = models.data?.data?.[0]?.id
  const origin = window.location.origin
  const snippetInput = { origin, apiKey: issued.key, model }

  return (
    <div className="flex flex-col gap-3">
      <InfoBanner tone="warning">
        Copy this key now. It is shown once, and reopening this guide issues a
        new one in its place.
      </InfoBanner>
      <CopyField label="API key" value={issued.key} />
      {model === undefined ? (
        <p className="text-xs text-muted">
          No model is being served yet, so the snippets name{" "}
          <code>{SNIPPET_MODEL_PLACEHOLDER}</code>. Replace it with a model from
          the{" "}
          <Link
            to="/models"
            className="font-medium text-link hover:text-link-hover"
          >
            Models
          </Link>{" "}
          page.
        </p>
      ) : null}
      <CopyField
        label="curl"
        value={buildCurlSnippet(snippetInput)}
        multiline
      />
      <CopyField
        label="Python (OpenAI SDK)"
        value={buildPythonSnippet(snippetInput)}
        multiline
      />
    </div>
  )
}

/**
 * The waiting state, and a failed attempt reported inside it.
 *
 * A failure dresses this row rather than replacing it: the guide is still
 * watching for the next request, so the shape of the panel does not change and
 * "Check now" stays where it was.
 */
function ListeningRow({
  attempt,
  isChecking,
  checkFailed,
  onCheckNow,
}: {
  attempt?: ActivationAttempt
  isChecking: boolean
  checkFailed: boolean
  onCheckNow: () => void
}) {
  const failure = attempt ? setupFailureCopy(attempt.error_category) : undefined

  return (
    <div
      className={`flex flex-wrap items-center justify-between gap-3 rounded-xl border px-4 py-3 ${
        failure
          ? "border-danger bg-danger-subtle"
          : "border-border bg-surface-alt"
      }`}
    >
      <div className="flex min-w-0 flex-col gap-0.5" role="status">
        {failure ? (
          <>
            <span className="text-sm font-medium text-danger">
              Request failed: {failure.cause}
            </span>
            <span className="text-xs text-muted">
              Still listening. Fix it and send the request again.
              {attempt?.occurred_at
                ? ` Last attempt ${formatRelative(attempt.occurred_at)}.`
                : ""}
            </span>
            {failure.hint ? (
              <Link
                to={failure.hint.to}
                className="text-xs font-medium text-link hover:text-link-hover"
              >
                {failure.hint.label}
              </Link>
            ) : null}
          </>
        ) : (
          <>
            <span className="text-sm font-medium text-foreground">
              {checkFailed
                ? "The gateway could not be checked"
                : "Listening for your first request"}
            </span>
            <span className="text-xs text-muted">
              {checkFailed
                ? "Leave this page open and try again."
                : "This page notices it within a few seconds."}
            </span>
          </>
        )}
      </div>
      <Button
        variant="outline"
        size="sm"
        isPending={isChecking}
        onPress={onCheckNow}
      >
        Check now
      </Button>
    </div>
  )
}

/**
 * The payoff, shown once, in place of the guide.
 *
 * The receipt is one mono line rather than a row of tiles: at this moment the
 * numbers are proof that the call was observed, not something to analyze, and
 * the pages that do analyze them are one click away.
 */
function SetupComplete({
  attempt,
  onDone,
}: {
  attempt?: ActivationAttempt
  onDone: () => void
}) {
  const navigate = useNavigate()
  const receipt = [
    attempt?.model,
    attempt?.latency_ms != null ? `${Math.round(attempt.latency_ms)} ms` : null,
    attempt?.cost_usd != null ? formatCost(attempt.cost_usd) : null,
  ].filter(Boolean)

  return (
    <Card>
      <Card.Content className="flex flex-wrap items-center justify-between gap-4 p-6">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-heading">Your first request went through</h2>
          <p className="text-sm text-muted">
            This workspace is serving traffic. Usage, spend and the activity log
            fill in from here.
          </p>
          {receipt.length > 0 ? (
            <p className="font-mono text-xs text-muted">
              {receipt.join(" · ")}
            </p>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="ghost" onPress={onDone}>
            Dismiss
          </Button>
          {/* Scoped to gateway traffic, like the pricing alarm's link: imported
              usage is somebody else's requests, and the one that just landed is
              the newest row of what is left. */}
          <Button
            variant="primary"
            onPress={() =>
              navigate({ to: "/activity", search: { source: "gateway" } })
            }
          >
            Open the activity log
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}
