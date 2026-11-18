import type { McpServerFoldCandidate } from "@/client"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { foldEffects, MAX_FOLD_COPIES } from "@/features/tools/mcpFold"

/**
 * The workspaces' own copies of a server an organization server can take the
 * place of, one box each, with what folding it would do to the people using
 * it. Ticked copies are folded; the rest stay as they are. Past
 * `MAX_FOLD_COPIES` ticked, the step that holds it must not submit.
 */
export function McpFoldChoices({
  candidates,
  selected,
  onChange,
  label = "Workspace copies",
  isDisabled = false,
}: {
  candidates: readonly McpServerFoldCandidate[]
  /** The ids of the ticked copies. */
  selected: readonly string[]
  onChange: (selected: string[]) => void
  /** The list's accessible name, apart per server where a step lists several. */
  label?: string
  isDisabled?: boolean
}) {
  const toggle = (serverId: string, isSelected: boolean) =>
    onChange(
      isSelected
        ? [...selected.filter((id) => id !== serverId), serverId]
        : selected.filter((id) => id !== serverId),
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
        {candidates.map((candidate) => (
          <li key={candidate.server_id}>
            <Checkbox
              ariaLabel={`${candidate.workspace_name}: ${candidate.name}`}
              isSelected={selected.includes(candidate.server_id)}
              isDisabled={isDisabled}
              // A 44px target, the box still beside the first line.
              className="min-h-11"
              onChange={(isSelected) => toggle(candidate.server_id, isSelected)}
            >
              <span className="flex min-w-0 flex-col gap-1">
                <span className="break-words">
                  <span className="font-medium">
                    {candidate.workspace_name}
                  </span>
                  : {candidate.name}
                </span>
                {foldEffects(candidate).map((line) => (
                  <span key={line} className="text-caption">
                    {line}
                  </span>
                ))}
              </span>
            </Checkbox>
          </li>
        ))}
      </ul>
    </div>
  )
}
