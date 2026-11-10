import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { Chip } from "@/design-system/indicators/Chip"
import { Dot } from "@/design-system/indicators/Dot"
import { McpCatalogBrowser } from "@/features/tools/McpCatalogBrowser"
import {
  type AddOutcome,
  type AddTone,
  addedOutcome,
  type CatalogPick,
  endpointKey,
  freeName,
  refusedOutcome,
} from "@/features/tools/mcpCatalog"
import { useFeaturedMcpServers } from "@/shared/api/mcpCatalog"
import { useCreateWorkspaceMcpServer } from "@/shared/api/tools"

// Adding servers from the catalog: pick, review, add. Each pick becomes a
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

export function McpCatalogDialog({
  isOpen,
  onOpenChange,
  workspaceId,
  workspaceName,
  existing,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  workspaceId: string
  workspaceName: string
  /** The workspace's servers, to mark what is already there and avoid its names. */
  existing: WorkspaceMcpServer[]
}) {
  const featured = useFeaturedMcpServers(isOpen)
  const create = useCreateWorkspaceMcpServer()

  const [step, setStep] = useState<Step>("browse")
  const [picks, setPicks] = useState<CatalogPick[]>([])
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [results, setResults] = useState<Result[]>([])
  const [isAdding, setIsAdding] = useState(false)

  const added = new Set(existing.map((server) => endpointKey(server.url)))
  // A featured entry and its catalog row are one server, so picking either
  // replaces the other. They can list different URLs once the catalog moved
  // on, and only the row that was ticked shows as picked: its URL is the one
  // that gets added.
  const isPicked = (pick: CatalogPick) =>
    picks.some((other) => other.name === pick.name && other.url === pick.url)

  const toggle = (pick: CatalogPick, isSelected: boolean) =>
    setPicks((current) =>
      isSelected
        ? [...current.filter((other) => other.name !== pick.name), pick]
        : current.filter((other) => other.name !== pick.name),
    )

  const review = () => {
    const taken = new Set(existing.map((server) => server.name))
    setDrafts(
      picks.map((pick) => {
        const name = freeName(pick.title, taken)
        taken.add(name)
        return { pick, name, token: "" }
      }),
    )
    setStep("review")
  }

  const editDraft = (name: string, change: Partial<Draft>) =>
    setDrafts((current) =>
      current.map((draft) =>
        draft.pick.name === name ? { ...draft, ...change } : draft,
      ),
    )

  const names = drafts.map((draft) => draft.name.trim())
  const nameProblem = (draft: Draft): string | undefined => {
    const name = draft.name.trim()
    if (!name) return "Give the server a name."
    if (names.filter((other) => other === name).length > 1) {
      return "Two servers in this list have this name."
    }
    return undefined
  }
  const canAdd = drafts.length > 0 && drafts.every((d) => !nameProblem(d))

  // One at a time, in order: the gateway locks the workspace per create, and a
  // refusal of one server (a name in use, the workspace's cap) must not stop
  // the ones after it from being tried and reported.
  const addAll = async () => {
    setIsAdding(true)
    const outcomes: Result[] = []
    for (const draft of drafts) {
      const name = draft.name.trim()
      try {
        const server = await create.mutateAsync({
          workspaceId,
          body: {
            name,
            url: draft.pick.url,
            authorization_token: draft.token || undefined,
            enabled: true,
          },
        })
        outcomes.push({ name, outcome: addedOutcome(server) })
      } catch (error) {
        outcomes.push({ name, outcome: refusedOutcome(errorMessage(error)) })
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
        title={`Review ${drafts.length} ${drafts.length === 1 ? "server" : "servers"}`}
        description={`Every key in ${workspaceName} may use a server once it is added. Change that on the server's page.`}
        footerStart={
          isAdding
            ? `Adding ${results.length + 1} of ${drafts.length}`
            : undefined
        }
        actions={
          <>
            <Button isDisabled={isAdding} onPress={() => setStep("browse")}>
              Back
            </Button>
            <Button
              variant="primary"
              isDisabled={!canAdd}
              isPending={isAdding}
              onPress={() => void addAll()}
            >
              {`Add ${drafts.length} ${drafts.length === 1 ? "server" : "servers"}`}
            </Button>
          </>
        }
      >
        {drafts.map((draft) => {
          const problem = nameProblem(draft)
          return (
            <DialogSection key={draft.pick.name}>
              <Field
                label="Name"
                value={draft.name}
                isRequired
                isDisabled={isAdding}
                isInvalid={problem !== undefined}
                errorMessage={problem}
                onChange={(name) => editDraft(draft.pick.name, { name })}
              />
              <code className="font-mono text-caption break-all">
                {draft.pick.url}
              </code>
              {draft.pick.requiresToken ? (
                <SecretField
                  label={`Authorization token for ${draft.name.trim() || draft.pick.title}`}
                  placeholder="Paste a token"
                  description="This server asks for a token. Leave it blank to add the server now and set the token later."
                  value={draft.token}
                  isDisabled={isAdding}
                  onChange={(token) => editDraft(draft.pick.name, { token })}
                />
              ) : null}
            </DialogSection>
          )
        })}
      </Dialog>
    )
  }

  return (
    <Dialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title="Browse catalog"
      description={`Remote MCP servers from the MCP Registry. Pick the ones to add to ${workspaceName}.`}
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
            Review
          </Button>
        </>
      }
    >
      <DialogSection>
        <ErrorBanner error={featured.error} />
        <McpCatalogBrowser
          featured={featured.data?.data ?? []}
          emptyCatalog="The catalog is empty. An operator can fill it from the MCP Registry on the MCP catalog page. You can still add a server by its URL."
          renderControl={(pick) => (
            <Checkbox
              ariaLabel={`Select ${pick.title}`}
              hasTouchTarget
              isSelected={isPicked(pick)}
              isDisabled={added.has(endpointKey(pick.url))}
              onChange={(isSelected) => toggle(pick, isSelected)}
            >
              {null}
            </Checkbox>
          )}
          renderNote={(pick) =>
            added.has(endpointKey(pick.url)) ? (
              <Chip tone="neutral">Already added</Chip>
            ) : null
          }
        />
      </DialogSection>
    </Dialog>
  )
}
