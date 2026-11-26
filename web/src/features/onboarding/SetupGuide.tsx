import { useNavigate } from "@tanstack/react-router"
import { useCallback, useEffect, useRef, useState } from "react"

import type { ActivationIssuedKey, WorkspaceActivation } from "@/client"
import { SetupSheet } from "@/features/onboarding/SetupSheet"
import { setupFailureCopy } from "@/features/onboarding/setupFailureCopy"
import { canManageWorkspace } from "@/features/organization/roles"
import {
  useCreateActivationKey,
  useDismissActivation,
  useWorkspaceActivation,
} from "@/shared/api/activation"
import { useModels } from "@/shared/api/models"
import { useOrganizationContext } from "@/shared/api/organizations"
import { resolveSnippetBaseUrl } from "@/shared/helpers/requestSnippets"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { usePrefersReducedMotion } from "@/shared/hooks/usePrefersReducedMotion"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

/**
 * The step between "a provider is configured" and "this dashboard has something
 * to show": a workspace's first successful request.
 *
 * **A sheet over the page, not a panel on it.** A dashboard whose workspace has
 * never served a request has nothing on it worth reading, so the guide takes
 * the screen the way the platform's flow did, and is offered once: every way
 * out, Skip or a link to another page, retires it for whoever it was offered
 * to (the workspace, or one member). It composes with the Overview's own
 * getting-started panel rather than competing with it: that one is shown while
 * the gateway has no provider, when a request cannot succeed yet, and this one
 * takes over once one exists.
 *
 * **Offered on either Overview.** Who may be offered it is the server's answer
 * (`experience_eligible`): whoever manages the workspace, for the workspace's
 * first request (on a multi-tenant deployment a tenant, and not the person who
 * operates the deployment), and a member who may run requests there, for their
 * own. Member Home holds its own first-run band back while it is offered.
 *
 * Mounted from the Overview rather than from the shell, which is where
 * `canServeRequests` is answered and where a new operator lands.
 */
export function SetupGuide({
  canServeRequests,
}: {
  /**
   * Whether a request from this caller could succeed, which is the page's
   * answer to give rather than this component's to fetch. Each Overview reads
   * it off what it may see: the operator's from `/providers`, which refuses a
   * tenant, and the organization's from the model catalog, which lists the
   * selectors that caller may name.
   *
   * Not a query of its own in here, and that is load-bearing: a page that
   * decides whether to render at all from the *fetching* state of the same
   * query would see a second observer inside a child re-trigger it on mount,
   * flip back to its loading branch, and unmount the observer that asked, which
   * remounts and asks again. One query, one owner.
   */
  canServeRequests: boolean
}) {
  const { selected } = useSelectedWorkspace()
  const workspaceId = selected?.workspace_id ?? null
  // Held back until the deployment can actually serve a request: with nothing to
  // route to this would be handing out a key for a call that cannot succeed, and
  // on the operator page the getting-started panel is the right guide instead.
  const activation = useWorkspaceActivation(workspaceId, canServeRequests)
  const context = useOrganizationContext().data
  // Offered for their own first request in a workspace other people already
  // use, so the sheet leaves out what only holds for its managers.
  const isMember =
    context !== undefined && !canManageWorkspace(context, selected?.role)

  // Checked here too, not only through `enabled`: another reader of the same
  // query (the update-check prompt) can fill the cache this observer sees.
  if (!canServeRequests || !workspaceId || !activation.data) {
    return null
  }

  return (
    // Keyed on the workspace, which is what discards the guide's own state when
    // the switcher moves: the issued key belongs to one workspace, and so do
    // "this session saw the offer" and "somebody closed the sheet".
    <SetupFlow
      key={workspaceId}
      workspaceId={workspaceId}
      workspaceName={selected?.name}
      isMember={isMember}
      activation={activation.data}
      checkFailed={activation.isError}
      onCheckNow={() => activation.refetch()}
    />
  )
}

function keyFingerprint({ key_prefix, key_suffix }: ActivationIssuedKey) {
  return `${key_prefix ?? ""}••${key_suffix ?? ""}`
}

function SetupFlow({
  workspaceId,
  workspaceName,
  isMember,
  activation: data,
  checkFailed,
  onCheckNow,
}: {
  workspaceId: string
  workspaceName?: string
  isMember: boolean
  activation: WorkspaceActivation
  checkFailed: boolean
  /** Resolves when the re-check lands, so the control can report it. */
  onCheckNow: () => Promise<unknown>
}) {
  const navigate = useNavigate()
  const prefersReducedMotion = usePrefersReducedMotion()
  const createKey = useCreateActivationKey()
  const dismiss = useDismissActivation()
  const models = useModels(true, workspaceId)
  const { recordEvent } = useTelemetry()
  const baseUrl = resolveSnippetBaseUrl()

  // Every way out retires the guide on the server; this hides the sheet while
  // that lands, including on the way to a page a guidance link names.
  const [isClosed, setIsClosed] = useState(false)
  const [isFinished, setIsFinished] = useState(false)
  // Only a press somebody made, never the background poll: `isFetching` would
  // put the control in its pending state every few seconds on its own.
  const [isChecking, setIsChecking] = useState(false)
  // Someone who left after taking a key and came back keeps it unless they ask
  // for a new one: minting rotates it, and the old one may already be pasted
  // somewhere.
  // Held as the fingerprint chosen, because a rotation elsewhere keeps the row
  // and changes the key: the choice no longer stands once it stops matching.
  const [keptKey, setKeptKey] = useState<string | null>(null)
  // The flow only celebrates a first request it was present for. Without this
  // latch, somebody who never opened the guide would be congratulated on the
  // traffic they already had, on the next page load after it arrived.
  const wasOffered = useRef(false)

  const isActivated = data.status === "activated"
  const isOffered = data.experience_eligible && !isClosed && !isActivated
  // Recorded after the commit rather than during the render that decided it.
  // Render has to stay pure, and the React Compiler is what makes that more
  // than a principle here: a render React discards would leave the latch set,
  // and a workspace that activated later would then be congratulated on an
  // offer no one was ever shown. The sheet and the payoff are never the same
  // render (`experience_eligible` goes false as `status` becomes activated), so
  // the effect has always run by the time the branch below reads it.
  // The same commit is where the offer is counted as shown, once per mount.
  useEffect(() => {
    if (!isOffered || wasOffered.current) return
    wasOffered.current = true
    recordEvent(TELEMETRY_EVENTS.ACTIVATION_GUIDE_SHOWN, {
      workspace_id: workspaceId,
    })
  }, [isOffered, workspaceId, recordEvent])
  // One showing is retired once, whichever way out lands first: a poll that
  // finds the first request can arrive while a Skip is still in flight.
  const retired = useRef(false)
  const recordRetired = useCallback(
    (reason: "skipped" | "left" | "activated") => {
      if (retired.current) return
      retired.current = true
      recordEvent(TELEMETRY_EVENTS.ACTIVATION_GUIDE_DISMISSED, {
        workspace_id: workspaceId,
        reason,
      })
    },
    [workspaceId, recordEvent],
  )
  // A first request landing while the sheet was up retires it too.
  useEffect(() => {
    if (isActivated && wasOffered.current) recordRetired("activated")
  }, [isActivated, recordRetired])

  if (isFinished) return null

  // The payoff is the sheet's second face, so a request that lands while the
  // sheet is up keeps it mounted rather than swapping in another dialog.
  const hasLanded = isActivated && wasOffered.current
  if (!isOffered && !hasLanded) return null

  // Recorded once the server has retired the guide, so a Skip that fails and
  // is pressed again is one dismissal rather than two.
  const skip = () =>
    dismiss.mutate(workspaceId, { onSuccess: () => recordRetired("skipped") })

  // Recorded on the press: the sheet closes at once and the link navigates.
  const leave = () => {
    setIsClosed(true)
    recordRetired("left")
    dismiss.mutate(workspaceId)
  }

  const checkNow = async () => {
    setIsChecking(true)
    try {
      // Match the original activation flow: let the working orb complete a beat.
      await Promise.all([
        onCheckNow(),
        new Promise((resolve) =>
          setTimeout(resolve, prefersReducedMotion ? 0 : 2_400),
        ),
      ])
    } finally {
      setIsChecking(false)
    }
  }

  return (
    <SetupSheet
      workspaceName={workspaceName}
      apiKey={createKey.data?.key}
      keyPrefix={createKey.data?.key_prefix ?? undefined}
      keySuffix={createKey.data?.key_suffix ?? undefined}
      // The mint refreshes this state, so a key minted here is described by
      // both; the plaintext in hand is the one to show.
      issuedKey={createKey.data ? undefined : (data.issued_key ?? undefined)}
      isUsingIssuedKey={
        data.issued_key != null && keptKey === keyFingerprint(data.issued_key)
      }
      onUseIssuedKey={() =>
        setKeptKey(data.issued_key ? keyFingerprint(data.issued_key) : null)
      }
      baseUrl={baseUrl}
      // The first model the gateway can serve, so the examples are runnable as
      // pasted. With none the placeholder stands and the sheet says what to do.
      model={models.data?.data?.[0]?.id}
      failure={
        data.status === "failed" && data.latest_attempt
          ? setupFailureCopy(data.latest_attempt.error_category)
          : undefined
      }
      isMember={isMember}
      attemptAt={
        data.status === "failed" ? data.latest_attempt?.occurred_at : undefined
      }
      isChecking={isChecking}
      checkFailed={checkFailed}
      keyError={createKey.error}
      isCreatingKey={createKey.isPending}
      skipError={dismiss.error}
      isSkipping={dismiss.isPending}
      onCreateKey={() => createKey.mutate(workspaceId)}
      onCheckNow={() => void checkNow()}
      onSkip={skip}
      onLeave={leave}
      landed={
        hasLanded
          ? {
              attempt: data.activation_attempt ?? undefined,
              onDismiss: () => setIsFinished(true),
              onOpenActivity: () => {
                setIsFinished(true)
                void navigate({
                  to: "/requests",
                  search: { source: "gateway" },
                })
              },
            }
          : undefined
      }
    />
  )
}
