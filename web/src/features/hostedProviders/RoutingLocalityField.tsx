import { Select, type SelectOption } from "@/design-system/forms/Select"

/** A locality the API accepts, or empty for undeclared. */
export type RoutingLocalityValue = "" | "local" | "private" | "cloud"

export const ROUTING_LOCALITY_OPTIONS: readonly SelectOption[] = [
  { value: "", label: "Not declared" },
  {
    value: "local",
    label: "Local",
    description: "Runs on the machine or network Otari runs on.",
  },
  {
    value: "private",
    label: "Private",
    description: "Runs on infrastructure you control.",
  },
  {
    value: "cloud",
    label: "Cloud",
    description: "Runs on a third party's service.",
  },
]

/** The label a provider page shows for a stored locality. */
export function routingLocalityLabel(value: string | null | undefined): string {
  return (
    ROUTING_LOCALITY_OPTIONS.find((option) => option.value === (value ?? ""))
      ?.label ?? "Not declared"
  )
}

/** The form value of a stored locality; one this build does not know reads as undeclared. */
export function routingLocalityValue(
  value: string | null | undefined,
): RoutingLocalityValue {
  return value === "local" || value === "private" || value === "cloud"
    ? value
    : ""
}

/**
 * Where the operator declares this upstream runs its inference, as a routing
 * policy with eligibility reads it. Every region inherits it. Undeclared, the
 * provider serves no such policy.
 */
export function RoutingLocalityField({
  value,
  onChange,
}: {
  value: RoutingLocalityValue
  onChange: (value: RoutingLocalityValue) => void
}) {
  return (
    <Select
      label="Routing locality"
      value={value}
      onChange={(next) => onChange(routingLocalityValue(next))}
      options={ROUTING_LOCALITY_OPTIONS}
      placeholder="Not declared"
      description="Optional. Where this upstream runs inference, for routing policies that restrict it. Every region inherits it. A provider with no declaration serves no such policy."
    />
  )
}
