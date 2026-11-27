import { Select } from "@/design-system/forms/Select"
import { McpInfoHint } from "@/features/tools/McpInfoHint"
import {
  AVAILABILITY,
  AVAILABILITY_OPTIONS,
  AVAILABILITY_ORDER,
  type McpAvailability,
} from "@/features/tools/mcpAvailability"

/**
 * The heading's Availability choice on a server's page, a workspace's or the
 * organization's. The page owns the write, so it also says when one failed.
 */
export function McpAvailabilitySelect({
  value,
  onChange,
  isDisabled,
}: {
  value: McpAvailability
  onChange: (next: McpAvailability) => void
  isDisabled: boolean
}) {
  return (
    <Select
      label="Availability"
      value={value}
      onChange={(next) => onChange(next as McpAvailability)}
      options={AVAILABILITY_OPTIONS}
      labelAction={
        <McpInfoHint label="Explain the three choices">
          {AVAILABILITY_ORDER.map((choice) => (
            <span key={choice}>
              <span className="font-medium">{AVAILABILITY[choice].label}:</span>{" "}
              {AVAILABILITY[choice].help}
            </span>
          ))}
        </McpInfoHint>
      }
      isDisabled={isDisabled}
      // No message line under it, so its edge is the row's edge.
      shouldReserveMessage={false}
      className="w-44"
    />
  )
}
