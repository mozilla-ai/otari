import { useState } from "react"

import type { McpCredentialMode, WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { RadioGroup, type RadioOption } from "@/design-system/forms/RadioGroup"
import { SecretField } from "@/design-system/forms/SecretField"
import { Dot } from "@/design-system/indicators/Dot"
import { McpCatalogBrowser } from "@/features/tools/McpCatalogBrowser"
import { McpCatalogSyncNotice } from "@/features/tools/McpCatalogSyncNotice"
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
  findFreeName,
  normalizeEndpoint,
  suggestMode,
} from "@/features/tools/mcpCatalog"
import {
  CREDENTIAL_MODE_COPY,
  SIGN_IN_MODE_LABEL,
  SIGN_IN_MODES,
} from "@/features/tools/mcpServerForm"
import { isMcpCatalogSyncing, useMcpCatalogSync } from "@/shared/api/mcpCatalog"
import {
  useCreateWorkspaceMcpServer,
  useMcpServerProbes,
} from "@/shared/api/tools"

// Adding servers from the catalog: pick, confirm, see what was added. Each pick becomes a
// stored server through the same create a pasted URL goes through, one request
// per server, so the URL check, the workspace's cap and the unique-name rule
// are the gateway's and every row reports what the gateway said about it.
//
// The catalog's description is shown and never stored. A purpose hint is
// prepended to the system message, and text from a public registry does not
// belong there unread.

type Step = "browse" | "review" | "results"

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

export function McpCatalogDialog({
  isOpen,
  onOpenChange,
  workspaceId,
  workspaceName,
  existing,
  onAddByUrl,
  initialPicks,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  workspaceId: string
  workspaceName: string
  /** The workspace's servers, to leave out what is already there and avoid its names. */
  existing: WorkspaceMcpServer[]
  /** Open the form that adds a server by URL, for one the catalog does not list. */
  onAddByUrl?: () => void
  /** Servers to open the confirm step on, skipping the browse: the one a member's request names. */
  initialPicks?: CatalogPick[]
}) {
  const sync = useMcpCatalogSync(isOpen)
  const create = useCreateWorkspaceMcpServer()

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

  const added = new Set(existing.map((server) => normalizeEndpoint(server.url)))
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
    setStep("review")
  }

  const editDraft = (name: string, change: Partial<Draft>) =>
    setDrafts((current) =>
      current.map((draft) =>
        draft.pick.name === name ? { ...draft, ...change } : draft,
      ),
    )

  const suggestionFor = (index: number, draft: Draft) => {
    const probe = probes[index]
    return suggestMode(draft.pick, probe?.data, probe?.isPending ?? false)
  }
  const modeOf = (index: number, draft: Draft): McpCredentialMode =>
    draft.mode ?? suggestionFor(index, draft).mode

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
  const addAll = async () => {
    setIsAdding(true)
    // Settled before the first create, so a probe landing mid-add changes nothing.
    const modes = drafts.map((draft, index) => modeOf(index, draft))
    const outcomes: Result[] = []
    for (const [index, draft] of drafts.entries()) {
      const name = draft.name.trim()
      const mode = modes[index]
      // A pasted token often ends in a newline, which would be stored and then refused.
      const token = mode === "shared" ? draft.token.trim() : ""
      try {
        const server = await create.mutateAsync({
          workspaceId,
          body: {
            name,
            url: draft.pick.url,
            ...(token ? { authorization_token: token } : {}),
            credential_mode: mode,
            ...availabilityFields(draft.availability),
          },
        })
        outcomes.push({ name, outcome: describeAdded(server) })
      } catch (error) {
        outcomes.push({ name, outcome: describeRefusal(errorMessage(error)) })
      }
      setResults([...outcomes])
    }
    setIsAdding(false)
    setStep("results")
  }

  const addedCount = results.filter(
    (result) => result.outcome.tone !== "danger",
  ).length

  if (step === "results") {
    return (
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        size="lg"
        title={`${addedCount} of ${results.length} added to ${workspaceName}`}
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

  if (step === "review") {
    return (
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        isDismissable={!isAdding}
        size="lg"
        title={
          drafts.length === 1
            ? `Add ${drafts[0].pick.title} to ${workspaceName}`
            : `Add ${drafts.length} servers to ${workspaceName}`
        }
        description={`Every key in ${workspaceName} may use a server once it is added. Change that on the server's page.`}
        footerStart={
          isAdding
            ? `Adding ${results.length + 1} of ${drafts.length}`
            : undefined
        }
        actions={
          <>
            {startsOnReview ? (
              <Button isDisabled={isAdding} onPress={() => onOpenChange(false)}>
                Cancel
              </Button>
            ) : (
              <Button isDisabled={isAdding} onPress={() => setStep("browse")}>
                Back
              </Button>
            )}
            <Button
              variant="primary"
              isDisabled={!canAdd}
              isPending={isAdding}
              onPress={() => void addAll()}
            >
              Confirm
            </Button>
          </>
        }
      >
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
                  isDisabled={isAdding}
                  isInvalid={problem !== undefined}
                  errorMessage={problem}
                  onChange={(name) => editDraft(draft.pick.name, { name })}
                />
                <RadioGroup
                  label={SIGN_IN_MODE_LABEL}
                  orientation="horizontal"
                  value={mode}
                  options={MODES}
                  isDisabled={isAdding}
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
                    isDisabled={isAdding}
                    onChange={(token) => editDraft(draft.pick.name, { token })}
                  />
                ) : null}
                <RadioGroup
                  label="Availability"
                  value={draft.availability}
                  options={AVAILABILITY_OPTIONS}
                  isDisabled={isAdding}
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
            {hasServers ? (
              <McpCatalogBrowser
                workspaceId={workspaceId}
                serverCount={status.server_count}
                isSyncing={isMcpCatalogSyncing(status)}
                selected={picks}
                isAdded={(pick) => added.has(normalizeEndpoint(pick.url))}
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
