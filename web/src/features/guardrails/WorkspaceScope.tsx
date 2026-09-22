import type { OrganizationGuardrail, Workspace } from "@/client"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Select } from "@/design-system/forms/Select"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"

// One width for every select and the credential box in a guardrail entry, so
// the entry has a rhythm instead of an edge per control. Full width where the
// row stacks on a phone.
export const SELECT_SLOT = "w-full sm:w-48"

/** What an entry's scope reads as, without making the reader count rows. */
export function scopeLabel(
  guardrail: OrganizationGuardrail,
  workspaces: readonly Workspace[],
): string {
  if (guardrail.applies_to_all_workspaces) {
    return "Every workspace, including new ones"
  }
  const count = guardrail.workspace_ids.length
  if (count === 0) return "No workspaces yet"
  const named = guardrail.workspace_ids
    .map((id) => workspaces.find((workspace) => workspace.id === id)?.name)
    .filter((name): name is string => name !== undefined)
  return named.length === count ? named.join(", ") : `${count} workspaces`
}

/** The scope picker's two positions, which are the two the server stores. */
export function WorkspaceScope({
  // Names which entry's scope this is, because the card renders one of these
  // per row plus one in the add form and the workspace names repeat in all of
  // them. Without it every checkbox on the card is labelled "Alpha".
  scopeName,
  appliesEverywhere,
  selected,
  workspaces,
  disabled,
  variant = "filter",
  onEverywhere,
  onToggle,
}: {
  /** Names the workspace group, so a box reads as "Beta" inside "prompt-injection". */
  scopeName: string
  appliesEverywhere: boolean
  selected: readonly string[]
  workspaces: readonly Workspace[]
  disabled?: boolean
  /**
   * Which half of the pair the picker is. A row on this card is a dense line
   * of toolbar controls; the dialog is a form, where a caption label beside
   * the control would be the one thing on it reading differently.
   */
  variant?: "filter" | "form"
  onEverywhere: (value: boolean) => void
  onToggle: (workspaceId: string) => void
}) {
  const scopeOptions = [
    { value: "all", label: "Every workspace" },
    { value: "chosen", label: "Chosen workspaces" },
  ]
  return (
    <div className="flex flex-col gap-2">
      {variant === "form" ? (
        <Select
          label="Runs in"
          value={appliesEverywhere ? "all" : "chosen"}
          onChange={(next) => onEverywhere(next === "all")}
          options={scopeOptions}
          isDisabled={disabled}
          shouldReserveMessage={false}
        />
      ) : (
        <div className={SELECT_SLOT}>
          <FilterSelect
            label="Runs in"
            value={appliesEverywhere ? "all" : "chosen"}
            onChange={(next) => onEverywhere(next === "all")}
            options={scopeOptions}
            disabled={disabled}
            fullWidth
          />
        </div>
      )}
      {appliesEverywhere ? null : (
        // A named group rather than a per-box aria-label. Each box is labelled
        // by the workspace name a reader can see, and the group says which
        // guardrail those names belong to; an aria-label on the box would have
        // replaced the visible text for assistive tech instead of qualifying it.
        <fieldset aria-label={scopeName} className="flex flex-wrap gap-3">
          {workspaces.map((workspace) => (
            <span key={workspace.id} className="text-sm text-muted">
              <Checkbox
                isSelected={selected.includes(workspace.id)}
                isDisabled={disabled}
                onChange={() => onToggle(workspace.id)}
              >
                {workspace.name}
              </Checkbox>
            </span>
          ))}
          {workspaces.length === 0 ? (
            <span className="text-caption">
              No workspaces to choose from yet.
            </span>
          ) : null}
        </fieldset>
      )}
    </div>
  )
}
