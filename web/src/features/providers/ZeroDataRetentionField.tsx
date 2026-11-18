import { Checkbox } from "@/design-system/forms/Checkbox"

/**
 * Whether a provider is declared to keep no prompt or completion data. Shared
 * by the hosted provider dialogs and the organization's own provider keys.
 */
export function ZeroDataRetentionField({
  isSelected,
  onChange,
}: {
  isSelected: boolean
  onChange: (isSelected: boolean) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <Checkbox isSelected={isSelected} onChange={onChange}>
        Zero data retention
      </Checkbox>
      <span className="text-caption">
        Tick this when your agreement with the provider says it keeps no prompts
        or completions. The models page shows it beside this provider and lets
        people filter on it. Otari does not verify it.
      </span>
    </div>
  )
}
