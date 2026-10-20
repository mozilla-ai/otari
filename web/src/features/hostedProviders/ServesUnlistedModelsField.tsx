import { Checkbox } from "@/design-system/forms/Checkbox"

/**
 * Whether a hosted provider serves a model its roster does not list, as a
 * `config.yml` provider does. Shared by the add and edit dialogs.
 */
export function ServesUnlistedModelsField({
  isSelected,
  onChange,
}: {
  isSelected: boolean
  onChange: (isSelected: boolean) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <Checkbox isSelected={isSelected} onChange={onChange}>
        Serve models not on its list
      </Checkbox>
      <span className="text-caption">
        Any model a request names is sent to this provider, priced or not, as
        with a config.yml provider. A model switched off on its list is still
        refused.
      </span>
    </div>
  )
}
