import { useState } from "react"

import type {
  CreateWorkspaceMcpServerRequest,
  McpCredentialMode,
  McpServerFoldCandidate,
  WorkspaceMcpServer,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { RadioGroup, type RadioOption } from "@/design-system/forms/RadioGroup"
import { SecretField } from "@/design-system/forms/SecretField"
import { Dot } from "@/design-system/indicators/Dot"
import { McpCatalogBrowser } from "@/features/tools/McpCatalogBrowser"
import { McpCatalogSyncNotice } from "@/features/tools/McpCatalogSyncNotice"
import { McpFoldChoices } from "@/features/tools/McpFoldChoices"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import {
  AVAILABILITY_OPTIONS,
  availabilityFields,
  type McpAvailability,
} from "@/features/tools/mcpAvailability"
import {
  type AddOutcome,
  type AddTone,
  type CatalogPick,
  describeAdded,
  describeRefusal,
  describeUnconfirmed,
  findFreeName,
  isUnconfirmed,
  suggestMode,
} from "@/features/tools/mcpCatalog"
import {
  defaultFoldIds,
  foldableIds,
  MAX_FOLD_COPIES,
  offerEachCopyOnce,
  settleAtMost,
} from "@/features/tools/mcpFold"
import {
  ADD_TARGET_LABEL,
  addTargetOptions,
  CREDENTIAL_MODE_COPY,
  type McpAddTarget,
  SIGN_IN_MODE_LABEL,
  SIGN_IN_MODES,
} from "@/features/tools/mcpServerForm"
import { isMcpCatalogSyncing, useMcpCatalogSync } from "@/shared/api/mcpCatalog"
import {
  type McpServerCopiesOf,
  useCreateOrganizationMcpServer,
  useReadMcpServerCopies,
  useRefreshAfterOrganizationServer,
} from "@/shared/api/organizationMcpServers"
import {
  useCreateWorkspaceMcpServer,
  useMcpServerProbes,
} from "@/shared/api/tools"

// Adding servers from the catalog: pick, confirm, see what was added. Each pick becomes a
// stored server through the same create a pasted URL goes through, one request
// per server, so the URL check, the workspace's cap and the unique-name rule
// are the gateway's and every row reports what the gateway said about it. An
// organization owner or admin adds for the whole organization by default, and
// is first offered the workspaces' own copies of each pick to fold into it.
//
// The catalog's description is shown and never stored. A purpose hint is
// prepended to the system message, and text from a public registry does not
// belong there unread.

type Step = "browse" | "review" | "fold" | "results"

interface Draft {
  pick: CatalogPick
  name: string
  token: string
  /** Absent until the admin picks one, so the suggested mode applies. */
  mode: McpCredentialMode | undefined
  availability: McpAvailability
}

interface Result {
  name: string
  outcome: AddOutcome
}

/** The workspaces' own copies of one pick, offered to fold into the organization's server. */
interface Fold {
  pickName: string
  title: string
  /** Whether the organization's server will be on. */
  enabled: boolean
  candidates: McpServerFoldCandidate[]
  selected: string[]
}

/** How many picks' copies are read at once: picks persist across catalog pages, so a batch can be long. */
const COPY_READS_AT_ONCE = 4

const TONE_DOT: Record<AddTone, string> = {
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
}

/** "Add" for one pick, "Add 3 servers" for several: the step that follows says to which workspace. */
function addLabel(count: number): string {
  return count > 1 ? `Add ${count} servers` : "Add"
}

// Without the per-option descriptions: the choices sit side by side here, and
// the chosen one's description is the group's.
const MODES: RadioOption[] = SIGN_IN_MODES.map(({ value, label }) => ({
  value,
  label,
}))
// Where the deployment cannot run a sign-in, the gateway refuses a Personal server.
const MODES_WITHOUT_SIGN_IN: RadioOption[] = MODES.map((option) =>
  option.value === "per_user" ? { ...option, isDisabled: true } : option,
)

export function McpCatalogDialog({
  isOpen,
  onOpenChange,
  workspaceId,
  workspaceName,
  existing,
  onAddByUrl,
  initialPicks,
  signInUnavailable,
  canAddForOrganization = false,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  workspaceId: string
  workspaceName: string
  /** The workspace's servers, whose names a new one avoids. */
  existing: WorkspaceMcpServer[]
  /** Open the form that adds a server by URL, for one the catalog does not list. */
  onAddByUrl?: () => void
  /** Servers to open the confirm step on, skipping the browse: the one a member's request names. */
  initialPicks?: CatalogPick[]
  /** Why this deployment cannot sign anyone in to a server yet, as the gateway words it; absent when it can. */
  signInUnavailable?: string | null
  /**
   * Whether the caller may add the servers for the whole organization
   * instead: an organization owner or admin is asked, with the organization
   * preselected.
   */
  canAddForOrganization?: boolean
}) {
  const sync = useMcpCatalogSync(isOpen)
  const create = useCreateWorkspaceMcpServer()
  const createForOrganization = useCreateOrganizationMcpServer()
  const readCopies = useReadMcpServerCopies()
  const refreshAfterUnconfirmed = useRefreshAfterOrganizationServer()
  const [target, setTarget] = useState<McpAddTarget>("organization")
  // Once the choice was offered, the batch keeps the destination picked, even
  // if organization access ends: it is then refused rather than added to the
  // workspace, until the admin picks the workspace themselves.
  const [wasOffered, setWasOffered] = useState(canAddForOrganization)
  if (canAddForOrganization && !wasOffered) setWasOffered(true)
  const forOrganization = wasOffered && target === "organization"
  const lostOrganization = forOrganization && !canAddForOrganization
  const targetName = forOrganization ? "your organization" : workspaceName
  // Settled when the copies are read, so the fold step adds what it was shown.
  const [settledModes, setSettledModes] = useState<McpCredentialMode[]>([])
  const [folds, setFolds] = useState<Fold[]>([])
  const [isReading, setReading] = useState(false)
  const [readError, setReadError] = useState<unknown>()
  // Where the last batch went, which the results step names.
  const [addedTo, setAddedTo] = useState("")

  const startsOnReview = (initialPicks?.length ?? 0) > 0
  const draftsFor = (chosen: CatalogPick[]): Draft[] => {
    const taken = new Set(existing.map((server) => server.name))
    return chosen.map((pick) => {
      const name = findFreeName(pick.title, taken)
      taken.add(name)
      return {
        pick,
        name,
        token: "",
        mode: undefined,
        availability: "on_request" as const,
      }
    })
  }
  const [step, setStep] = useState<Step>(startsOnReview ? "review" : "browse")
  const [picks, setPicks] = useState<CatalogPick[]>(initialPicks ?? [])
  const [drafts, setDrafts] = useState<Draft[]>(() =>
    draftsFor(initialPicks ?? []),
  )
  const [results, setResults] = useState<Result[]>([])
  const [isAdding, setIsAdding] = useState(false)
  // Asked only once the review step lists the picks, and never waited on.
  const probes = useMcpServerProbes(
    workspaceId,
    step === "review" ? drafts.map((draft) => draft.pick.url) : [],
  )

  const isPicked = (pick: CatalogPick) =>
    picks.some((other) => other.name === pick.name)

  const toggle = (pick: CatalogPick, isSelected: boolean) =>
    setPicks((current) =>
      isSelected
        ? [...current.filter((other) => other.name !== pick.name), pick]
        : current.filter((other) => other.name !== pick.name),
    )

  const review = () => {
    setDrafts(draftsFor(picks))
    setReadError(undefined)
    setStep("review")
  }

  const editDraft = (name: string, change: Partial<Draft>) =>
    setDrafts((current) =>
      current.map((draft) =>
        draft.pick.name === name ? { ...draft, ...change } : draft,
      ),
    )

  const rawSuggestionFor = (index: number, draft: Draft) => {
    const probe = probes[index]
    return suggestMode(draft.pick, probe?.data, probe?.isPending ?? false)
  }
  /** The server asks each person to sign in, which this deployment cannot run. */
  const cannotSignIn = (index: number, draft: Draft) =>
    Boolean(signInUnavailable) &&
    rawSuggestionFor(index, draft).mode === "per_user"
  const suggestionFor = (index: number, draft: Draft) =>
    cannotSignIn(index, draft)
      ? { mode: "shared" as const, reason: "" }
      : rawSuggestionFor(index, draft)
  const modeOf = (index: number, draft: Draft): McpCredentialMode => {
    const chosen = draft.mode ?? suggestionFor(index, draft).mode
    // A choice made before the deployment said it cannot sign anyone in does not stand.
    return signInUnavailable && chosen === "per_user" ? "shared" : chosen
  }

  const names = drafts.map((draft) => draft.name.trim())
  const findNameProblem = (draft: Draft): string | undefined => {
    const name = draft.name.trim()
    if (!name) return "Give the server a name."
    if (names.filter((other) => other === name).length > 1) {
      return "Two servers in this list have this name."
    }
    return undefined
  }
  const canAdd =
    drafts.length > 0 && drafts.every((draft) => !findNameProblem(draft))

  // One at a time, in order: the gateway locks the workspace per create, and a
  // refusal of one server (a name in use, the workspace's cap) must not stop
  // the ones after it from being tried and reported.
  const addAll = async (
    modes: McpCredentialMode[],
    foldIds: Record<string, string[]>,
    toOrganization: boolean,
  ) => {
    setAddedTo(toOrganization ? "your organization" : workspaceName)
    setIsAdding(true)
    const outcomes: Result[] = []
    let unconfirmed = false
    for (const [index, draft] of drafts.entries()) {
      const name = draft.name.trim()
      const mode = modes[index]
      // A pasted token often ends in a newline, which would be stored and then refused.
      const token = mode === "shared" ? draft.token.trim() : ""
      const body: CreateWorkspaceMcpServerRequest = {
        name,
        url: draft.pick.url,
        ...(token ? { authorization_token: token } : {}),
        credential_mode: mode,
        ...availabilityFields(draft.availability),
      }
      const folded = foldIds[draft.pick.name]
      try {
        const server = toOrganization
          ? await createForOrganization.mutateAsync(
              folded ? { ...body, fold_server_ids: folded } : body,
            )
          : await create.mutateAsync({ workspaceId, body })
        outcomes.push({ name, outcome: describeAdded(server) })
      } catch (error) {
        const message = errorMessage(error)
        // A lost answer is not a refusal: the server, and its folded copies, may be stored.
        unconfirmed ||= isUnconfirmed(error)
        outcomes.push({
          name,
          outcome: isUnconfirmed(error)
            ? describeUnconfirmed(message)
            : describeRefusal(message),
        })
      }
      setResults([...outcomes])
    }
    if (unconfirmed) refreshAfterUnconfirmed()
    setIsAdding(false)
    setStep("results")
  }

  // For the whole organization, the workspaces' own copies of each pick are
  // read first, and offered to fold when there are any. The target is settled
  // here too, so the fold step adds where it was reviewed for.
  const confirm = async () => {
    // Settled before anything is stored, so a probe landing later changes nothing.
    const modes = drafts.map((draft, index) => modeOf(index, draft))
    if (!forOrganization) {
      await addAll(modes, {}, false)
      return
    }
    // Every pick is read again on each Confirm, the ones that answered last
    // time too: another admin may have added a copy since.
    const reads = drafts.map((draft, index) => {
      const of: McpServerCopiesOf = {
        url: draft.pick.url,
        credentialMode: modes[index],
        withToken: modes[index] === "shared" && draft.token.trim() !== "",
      }
      return { title: draft.name.trim() || draft.pick.title, of }
    })
    setReadError(undefined)
    setReading(true)
    const outcomes = await settleAtMost(reads, COPY_READS_AT_ONCE, (read) =>
      readCopies(read.of),
    )
    setReading(false)
    const read: McpServerFoldCandidate[][] = []
    const failed: { title: string; reason: unknown }[] = []
    for (const [index, outcome] of outcomes.entries()) {
      if (outcome.status === "fulfilled") {
        read.push(outcome.value)
      } else {
        failed.push({ title: reads[index].title, reason: outcome.reason })
      }
    }
    if (failed.length > 0) {
      setReadError(
        new Error(
          `Could not look for copies of ${failed
            .map((each) => each.title)
            .join(
              ", ",
            )}: ${errorMessage(failed[0].reason)} Nothing was added. Confirm looks again.`,
        ),
      )
      return
    }
    const found = offerEachCopyOnce(read)
    const withCopies = drafts.flatMap((draft, index) => {
      const { enabled } = availabilityFields(draft.availability)
      return found[index].length > 0
        ? [
            {
              pickName: draft.pick.name,
              title: reads[index].title,
              enabled,
              candidates: found[index],
              selected: defaultFoldIds(found[index], enabled),
            },
          ]
        : []
    })
    if (withCopies.length === 0) {
      await addAll(modes, {}, true)
      return
    }
    setSettledModes(modes)
    setFolds(withCopies)
    setStep("fold")
  }

  const reviewTitle =
    drafts.length === 1
      ? `Add ${drafts[0].pick.title} to ${targetName}`
      : `Add ${drafts.length} servers to ${targetName}`

  const addedCount = results.filter(
    (result) => result.outcome.tone !== "danger",
  ).length

  if (step === "results") {
    return (
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        size="lg"
        title={`${addedCount} of ${results.length} added to ${addedTo}`}
        description="Each server was checked as it was stored."
        actions={
          <Button variant="primary" onPress={() => onOpenChange(false)}>
            Done
          </Button>
        }
      >
        <DialogSection>
          <ul aria-label="What was added" className="flex flex-col gap-3">
            {results.map((result) => (
              <li key={result.name} className="flex items-start gap-3">
                <Dot className={`mt-1.5 ${TONE_DOT[result.outcome.tone]}`} />
                <div className="flex min-w-0 flex-col gap-1">
                  <span className="flex flex-wrap items-center gap-2">
                    <span className="font-medium break-words">
                      {result.name}
                    </span>
                    <span className="text-mono-caption text-muted">
                      {result.outcome.label.toUpperCase()}
                    </span>
                  </span>
                  <span className="text-sm text-muted">
                    {result.outcome.detail}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </DialogSection>
      </Dialog>
    )
  }

  if (step === "fold") {
    const foldIds = Object.fromEntries(
      folds.map((fold) => [
        fold.pickName,
        foldableIds(fold.candidates, fold.selected, fold.enabled),
      ]),
    )
    const isOverLimit = Object.values(foldIds).some(
      (ids) => ids.length > MAX_FOLD_COPIES,
    )
    return (
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        isDismissable={!isAdding}
        size="lg"
        title={reviewTitle}
        description="Some workspaces have their own server at the same URL. Pick the ones to fold into the organization's."
        footerStart={
          isAdding
            ? `Adding ${results.length + 1} of ${drafts.length}`
            : undefined
        }
        actions={
          <>
            <Button isDisabled={isAdding} onPress={() => setStep("review")}>
              Back
            </Button>
            <Button
              variant="primary"
              isDisabled={lostOrganization || isOverLimit}
              isPending={isAdding}
              onPress={() => void addAll(settledModes, foldIds, true)}
            >
              Confirm
            </Button>
          </>
        }
      >
        {/* Organization access ended while the step was open: nothing is added anywhere else instead. */}
        {lostOrganization ? (
          <DialogSection>
            <InfoBanner tone="warning">
              You can no longer add servers for the whole organization. Go back
              to add them to {workspaceName} instead.
            </InfoBanner>
          </DialogSection>
        ) : null}
        {folds.map((fold) => (
          <DialogSection key={fold.pickName}>
            <section
              aria-label={`Copies of ${fold.title}`}
              className="flex flex-col gap-3"
            >
              <span className="font-medium break-words">{fold.title}</span>
              <McpFoldChoices
                label={`Workspace copies of ${fold.title}`}
                candidates={fold.candidates}
                selected={foldIds[fold.pickName]}
                targetEnabled={fold.enabled}
                isDisabled={isAdding || lostOrganization}
                onChange={(selected) =>
                  setFolds((current) =>
                    current.map((other) =>
                      other.pickName === fold.pickName
                        ? { ...other, selected }
                        : other,
                    ),
                  )
                }
              />
            </section>
          </DialogSection>
        ))}
      </Dialog>
    )
  }

  if (step === "review") {
    const isBusy = isAdding || isReading
    return (
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        isDismissable={!isBusy}
        size="lg"
        title={reviewTitle}
        description={
          forOrganization
            ? "Each server is added for the whole organization, and its access rules decide who can use it. Change them on the server's page."
            : `Every key in ${workspaceName} may use a server once it is added. Change that on the server's page.`
        }
        footerStart={
          isAdding
            ? `Adding ${results.length + 1} of ${drafts.length}`
            : undefined
        }
        actions={
          <>
            {startsOnReview ? (
              <Button isDisabled={isBusy} onPress={() => onOpenChange(false)}>
                Cancel
              </Button>
            ) : (
              <Button isDisabled={isBusy} onPress={() => setStep("browse")}>
                Back
              </Button>
            )}
            <Button
              variant="primary"
              isDisabled={!canAdd || lostOrganization}
              isPending={isBusy}
              onPress={() => void confirm()}
            >
              Confirm
            </Button>
          </>
        }
      >
        {wasOffered || readError ? (
          <DialogSection>
            <ErrorBanner error={readError} />
            {lostOrganization ? (
              <InfoBanner tone="warning">
                You can no longer add servers for the whole organization. Choose
                This workspace to add them to {workspaceName} instead.
              </InfoBanner>
            ) : null}
            {wasOffered ? (
              <RadioGroup
                label={ADD_TARGET_LABEL}
                value={target}
                options={addTargetOptions(workspaceName)}
                isDisabled={isBusy}
                onChange={(next) => setTarget(next as McpAddTarget)}
              />
            ) : null}
          </DialogSection>
        ) : null}
        {signInUnavailable ? (
          <DialogSection>
            <InfoBanner>{signInUnavailable}</InfoBanner>
          </DialogSection>
        ) : null}
        {drafts.map((draft, index) => {
          const problem = findNameProblem(draft)
          const mode = modeOf(index, draft)
          const suggestion = suggestionFor(index, draft)
          const label = draft.name.trim() || draft.pick.title
          return (
            <DialogSection key={draft.pick.name}>
              <section
                aria-label={draft.pick.title}
                className="flex flex-col gap-3"
              >
                <div className="flex items-center gap-3">
                  <McpServerLogo
                    name={draft.pick.title}
                    logoUrl={draft.pick.logoUrl}
                    tone={draft.pick.logoTone}
                  />
                  <code className="min-w-0 font-mono text-caption break-all">
                    {draft.pick.url}
                  </code>
                </div>
                <Field
                  label="Name"
                  value={draft.name}
                  isRequired
                  isDisabled={isBusy}
                  isInvalid={problem !== undefined}
                  errorMessage={problem}
                  onChange={(name) => editDraft(draft.pick.name, { name })}
                />
                {cannotSignIn(index, draft) ? (
                  <InfoBanner tone="warning">
                    {`${draft.pick.title} asks each person to sign in, which this deployment cannot do yet.`}
                  </InfoBanner>
                ) : null}
                <RadioGroup
                  label={SIGN_IN_MODE_LABEL}
                  orientation="horizontal"
                  value={mode}
                  options={signInUnavailable ? MODES_WITHOUT_SIGN_IN : MODES}
                  isDisabled={isBusy}
                  description={
                    draft.mode === undefined && suggestion.reason
                      ? `${CREDENTIAL_MODE_COPY[mode].help} ${suggestion.reason}`
                      : CREDENTIAL_MODE_COPY[mode].help
                  }
                  onChange={(next) =>
                    editDraft(draft.pick.name, {
                      mode: next as McpCredentialMode,
                    })
                  }
                />
                {mode === "shared" ? (
                  <SecretField
                    label={`Authorization token for ${label}`}
                    placeholder={
                      draft.pick.requiresToken ? "Paste a token" : "Optional"
                    }
                    description={
                      draft.pick.requiresToken
                        ? "This server asks for an account. Paste a token, or leave this blank and set the account up on the server's page after adding: you can sign in for everyone there."
                        : "Optional. To sign in for everyone instead, leave this blank and set the account up on the server's page after adding."
                    }
                    value={draft.token}
                    isDisabled={isBusy}
                    onChange={(token) => editDraft(draft.pick.name, { token })}
                  />
                ) : null}
                <RadioGroup
                  label="Availability"
                  value={draft.availability}
                  options={AVAILABILITY_OPTIONS}
                  isDisabled={isBusy}
                  onChange={(next) =>
                    editDraft(draft.pick.name, {
                      availability: next as McpAvailability,
                    })
                  }
                />
              </section>
            </DialogSection>
          )
        })}
      </Dialog>
    )
  }

  const status = sync.data
  const hasServers = (status?.server_count ?? 0) > 0
  return (
    <Dialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title="Browse catalog"
      description={`Remote MCP servers listed in the MCP Registry. Pick the ones to add to ${workspaceName}.`}
      footerStart={
        picks.length > 0
          ? `${picks.length} selected`
          : "Listing a server here does not vouch for it."
      }
      actions={
        <>
          <Button onPress={() => onOpenChange(false)}>Cancel</Button>
          <Button
            variant="primary"
            isDisabled={picks.length === 0}
            onPress={review}
          >
            {addLabel(picks.length)}
          </Button>
        </>
      }
    >
      <DialogSection>
        <ErrorBanner error={sync.error} />
        {status === undefined ? (
          sync.isPending ? (
            <EmptyMessage>Reading the catalog.</EmptyMessage>
          ) : null
        ) : (
          <div className="flex flex-col gap-4">
            <McpCatalogSyncNotice status={status} workspaceId={workspaceId} />
            {signInUnavailable ? (
              <InfoBanner>{signInUnavailable}</InfoBanner>
            ) : null}
            {hasServers ? (
              <McpCatalogBrowser
                workspaceId={workspaceId}
                serverCount={status.server_count}
                isSyncing={isMcpCatalogSyncing(status)}
                selected={picks}
                renderControl={(pick) => (
                  <Checkbox
                    ariaLabel={`Select ${pick.title}`}
                    hasTouchTarget
                    isSelected={isPicked(pick)}
                    onChange={(isSelected) => toggle(pick, isSelected)}
                  >
                    {null}
                  </Checkbox>
                )}
                onAddByUrl={onAddByUrl}
              />
            ) : null}
          </div>
        )}
      </DialogSection>
    </Dialog>
  )
}
