import { useNavigate } from "@tanstack/react-router"
import { useEffect, useRef, useState } from "react"

import type { ActivationApiKey, WorkspaceActivation } from "@/client"
import { SetupSheet } from "@/features/onboarding/SetupSheet"
import { SetupSuccess } from "@/features/onboarding/SetupSuccess"
import { setupFailureCopy } from "@/features/onboarding/setupFailureCopy"
import {
  useCreateActivationKey,
  useDismissActivation,
  useWorkspaceActivation,
} from "@/shared/api/activation"
import { useModels } from "@/shared/api/models"
import { resolveSnippetBaseUrl } from "@/shared/helpers/requestSnippets"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment, useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * The step between "a provider is configured" and "this dashboard has something
 * to show": a workspace's first successful request.
 *
 * **A sheet over the page, not a panel on it.** A dashboard whose workspace has
 * never served a request has nothing on it worth reading, so the guide takes
 * the screen the way the platform's flow did, offers a way out in its footer,
 * and never returns once that is taken. It composes with the Overview's own
 * getting-started panel rather than competing with it: that one is shown while
 * the gateway has no provider, when a request cannot succeed yet, and this one
 * takes over once one exists.
 *
 * **Offered on either Overview.** Who may be offered it is the server's answer
 * (`experience_eligible`, which ends in `has_workspace_management_access`), so
 * it goes to whoever manages the workspace: on a multi-tenant deployment a
 * tenant, and not the person who operates the deployment.
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
  const surfaces = useSurfaces()
  const { selected } = useSelectedWorkspace()
  const workspaceId = selected?.workspace_id ?? null
  // Held back until the deployment can actually serve a request: with nothing to
  // route to this would be handing out a key for a call that cannot succeed, and
  // on the operator page the getting-started panel is the right guide instead.
  const activation = useWorkspaceActivation(
    workspaceId,
    surfaces("workspaces") && canServeRequests,
  )

  if (!workspaceId || !activation.data) {
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
      activation={activation.data}
      checkFailed={activation.isError}
      onCheckNow={() => activation.refetch()}
    />
  )
}

function SetupFlow({
  workspaceId,
  workspaceName,
  activation: data,
  checkFailed,
  onCheckNow,
}: {
  workspaceId: string
  workspaceName?: string
  activation: WorkspaceActivation
  checkFailed: boolean
  /** Resolves when the re-check lands, so the control can report it. */
  onCheckNow: () => Promise<unknown>
}) {
  const navigate = useNavigate()
  const createKey = useCreateActivationKey()
  const dismiss = useDismissActivation()
  const models = useModels()
  // Where a request from this deployment belongs, which is not always the
  // address that served this page: a hosted control plane serves the dashboard
  // and not the API. Undefined when it has not said where its gateway is, and
  // the examples are then withheld rather than aimed at this host (otari#823).
  const baseUrl = resolveSnippetBaseUrl(useDeployment())

  // Closing the sheet without skipping: for this page load only, so Escape is
  // not a decision an operator cannot take back. Skipping is the permanent one,
  // and the server records it.
  // The issued key and a failed mint, held here rather than read off the
  // mutation. The effect below says why the observer cannot carry them.
  const [issued, setIssued] = useState<ActivationApiKey>()
  const [mintError, setMintError] = useState<unknown>()
  const [isClosed, setIsClosed] = useState(false)
  const [isFinished, setIsFinished] = useState(false)
  // Only a press somebody made, never the background poll: `isFetching` would
  // put the control in its pending state every few seconds on its own.
  const [isChecking, setIsChecking] = useState(false)
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
  useEffect(() => {
    if (isOffered) wasOffered.current = true
  }, [isOffered])

  // The key is minted when the sheet opens rather than on a press, because the
  // sheet *is* the press: it is the whole screen, it exists to hand a key over,
  // and asking for one more click before showing the thing the screen is about
  // is a step with nothing behind it. A ref rather than the mutation's own
  // state, so a second mint cannot be issued by a re-render or by React
  // remounting the effect; the component is keyed on the workspace, so
  // switching workspaces gets a fresh one.
  const hasRequestedKey = useRef(false)
  // Fired once across both of React's development effect invocations, and with
  // no "ignore a late result" flag, which is the part that is easy to get
  // wrong: such a flag is scoped to one invocation while the ref above is
  // scoped to the component, so the first invocation's cleanup would discard
  // the only request in flight and the second would decline to replace it,
  // having been told one was already sent. The key minted, arrived, and was
  // thrown away, leaving the sheet on "Creating your API key…" forever. In
  // development only, so a production build never showed it.
  //
  // Nothing is needed in its place. This component is keyed on the workspace,
  // so switching destroys the instance, and React discards a `setState` that
  // lands on one that is gone.
  //
  // `mutateAsync` rather than `mutate` with callbacks, because the result then
  // comes from the promise rather than through the mutation observer, which is
  // one less thing between the response and the screen.
  useEffect(() => {
    if (!isOffered || hasRequestedKey.current) return
    hasRequestedKey.current = true
    createKey.mutateAsync(workspaceId).then(setIssued, setMintError)
  }, [isOffered, workspaceId, createKey.mutateAsync])

  if (isFinished) return null

  if (isActivated) {
    return wasOffered.current ? (
      <SetupSuccess
        attempt={data.activation_attempt ?? undefined}
        onDismiss={() => setIsFinished(true)}
        onOpenActivity={() => {
          setIsFinished(true)
          void navigate({ to: "/activity", search: { source: "gateway" } })
        }}
      />
    ) : null
  }

  if (!isOffered) return null

  const checkNow = async () => {
    setIsChecking(true)
    try {
      await onCheckNow()
    } finally {
      setIsChecking(false)
    }
  }

  return (
    <SetupSheet
      workspaceName={workspaceName}
      apiKey={issued?.key}
      baseUrl={baseUrl}
      // The first model the gateway can serve, so the examples are runnable as
      // pasted. With none the placeholder stands and the sheet says what to do.
      model={models.data?.data?.[0]?.id}
      failure={
        data.status === "failed" && data.latest_attempt
          ? setupFailureCopy(data.latest_attempt.error_category)
          : undefined
      }
      attemptAt={
        data.status === "failed" ? data.latest_attempt?.occurred_at : undefined
      }
      isChecking={isChecking}
      checkFailed={checkFailed}
      keyError={mintError}
      skipError={dismiss.error}
      isSkipping={dismiss.isPending}
      onCheckNow={() => void checkNow()}
      onSkip={() => dismiss.mutate(workspaceId)}
      onDismiss={() => setIsClosed(true)}
    />
  )
}
