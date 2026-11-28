import { useId } from "react"

import type { McpCredentialMode, McpServerFoldCandidate } from "@/client"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import {
  foldBlocker,
  foldEffects,
  MAX_FOLD_COPIES,
} from "@/features/tools/mcpFold"

/**
 * The workspaces' own copies of a server an organization server can take the
 * place of, one box each, with what folding it would do to the people using
 * it. Ticked copies are folded; the rest stay as they are. A workspace folds
 * one copy, so ticking one unticks the others of its workspace, and a copy the
 * server cannot replace is listed with the reason and cannot be ticked. Past
 * `MAX_FOLD_COPIES` ticked, the step that holds it must not submit.
 */
export function McpFoldChoices({
  candidates,
  selected,
  onChange,
  targetEnabled,
  targetMode,
  label = "Workspace copies",
  isDisabled = false,
}: {
  candidates: readonly McpServerFoldCandidate[]
  /** The ids of the ticked copies. */
  selected: readonly string[]
  onChange: (selected: string[]) => void
  /** Whether the organization's server is on. */
  targetEnabled: boolean
  /** How the organization's server signs in. */
  targetMode: McpCredentialMode
  /** The list's accessible name, apart per server where a step lists several. */
  label?: string
  isDisabled?: boolean
}) {
  const idBase = useId()
  const perWorkspace = new Map<string, number>()
  for (const candidate of candidates) {
    perWorkspace.set(
      candidate.workspace_id,
      (perWorkspace.get(candidate.workspace_id) ?? 0) + 1,
    )
  }
  const workspaceOf = new Map(
    candidates.map((candidate) => [
      candidate.server_id,
      candidate.workspace_id,
    ]),
  )
  const toggle = (candidate: McpServerFoldCandidate, isSelected: boolean) =>
    onChange(
      isSelected
        ? [
            ...selected.filter(
              (id) => workspaceOf.get(id) !== candidate.workspace_id,
            ),
            candidate.server_id,
          ]
        : selected.filter((id) => id !== candidate.server_id),
    )
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-muted">
        These workspaces have their own server at the same URL. A copy you tick
        is replaced by the organization&rsquo;s server: its access rules become
        that workspace&rsquo;s settings there, and its id keeps working. A copy
        left unticked stays as it is.
      </p>
      {candidates.length > MAX_FOLD_COPIES ? (
        <InfoBanner
          tone={selected.length > MAX_FOLD_COPIES ? "warning" : "info"}
        >
          {`Up to ${MAX_FOLD_COPIES} copies can be folded at a time, and ${selected.length} are ticked.${
            selected.length > MAX_FOLD_COPIES ? " Untick some to continue." : ""
          }`}
        </InfoBanner>
      ) : null}
      <ul aria-label={label} className="flex flex-col gap-3">
        {candidates.map((candidate) => {
          const blocker = foldBlocker(candidate, targetEnabled)
          const isShared = (perWorkspace.get(candidate.workspace_id) ?? 0) > 1
          const effectsId = `${idBase}-${candidate.server_id}`
          return (
            <li key={candidate.server_id}>
              <Checkbox
                ariaLabel={`${candidate.workspace_name}: ${candidate.name}`}
                // What folding it does is read with its name, not only seen.
                ariaDescribedBy={effectsId}
                isSelected={
                  blocker === "" && selected.includes(candidate.server_id)
                }
                isDisabled={isDisabled || blocker !== ""}
                // A 44px target, the box still beside the first line.
                className="min-h-11"
                onChange={(isSelected) => toggle(candidate, isSelected)}
              >
                <span className="flex min-w-0 flex-col gap-1">
                  <span className="break-words">
                    <span className="font-medium">
                      {candidate.workspace_name}
                    </span>
                    : {candidate.name}
                  </span>
                  <span id={effectsId} className="flex flex-col gap-1">
                    {blocker ? (
                      <span className="text-caption text-warning">
                        {blocker}
                      </span>
                    ) : null}
                    {isShared ? (
                      <span className="text-caption">
                        Its workspace has another copy here. Only one of them
                        can be folded.
                      </span>
                    ) : null}
                    {foldEffects(candidate, targetMode).map((line) => (
                      <span key={line} className="text-caption">
                        {line}
                      </span>
                    ))}
                  </span>
                </span>
              </Checkbox>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
