import { SettingRow } from "@/design-system/layout/SettingRow"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"

/** Who a workspace pins to run a provider's own tool keyword, `null` where it pins nobody. */
export type Executor = "auto" | "otari" | "provider"

const UNPINNED = "default"

const OPTIONS = [
  { value: UNPINNED, label: "Not set" },
  { value: "auto", label: "Auto: provider when native, else Otari" },
  { value: "otari", label: "Always Otari" },
  { value: "provider", label: "Always the provider" },
]

/**
 * Who runs a provider's own tool keyword for the workspace's requests, so a client
 * written against that provider keeps working while Otari runs the tool.
 *
 * "Not set" leaves the request's header, and then the deployment, in charge.
 */
export function ExecutorRow({
  label,
  help,
  value,
  onChange,
  disabled,
}: {
  label: string
  help: string
  value: Executor | null | undefined
  onChange: (next: Executor | null) => void
  disabled: boolean
}) {
  return (
    <SettingRow
      label={label}
      help={help}
      control={
        <FilterSelect
          fullWidth
          ariaLabel={label}
          value={value ?? UNPINNED}
          onChange={(next) =>
            onChange(next === UNPINNED ? null : (next as Executor))
          }
          options={OPTIONS}
          disabled={disabled}
        />
      }
    />
  )
}
