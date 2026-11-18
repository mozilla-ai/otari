import { useState } from "react"

import { FormDialog } from "@/design-system/feedback/FormDialog"
import { MultiSelect } from "@/design-system/forms/MultiSelect"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  DATA_LOCATION_OPTIONS,
  dataLocationLabel,
} from "@/features/hostedProviders/dataLocations"

const LOCATION_OPTIONS = DATA_LOCATION_OPTIONS.map((option) => ({
  id: option.value,
  label: option.label,
  hint: option.hint,
}))

const SCOPE_OPTIONS = [
  { value: "any", label: "Any location" },
  { value: "only", label: "Only these locations" },
  {
    value: "none",
    label: "No location",
    description: "No hosted provider serves this workspace.",
  },
] as const

type Scope = (typeof SCOPE_OPTIONS)[number]["value"]

const scopeOf = (allowed: readonly string[] | null): Scope =>
  allowed === null ? "any" : allowed.length === 0 ? "none" : "only"

/** What a workspace's list reads as in a sentence: "Any location", or the names. */
export function describeAllowedDataLocations(
  allowed: readonly string[] | null | undefined,
): string {
  if (allowed == null) return "Any location"
  if (allowed.length === 0) return "No location"
  return allowed.map(dataLocationLabel).join(", ")
}

/**
 * Where a hosted provider may keep this workspace's request data: anywhere, or
 * only in the locations chosen. The European Union admits each member state.
 */
export function AllowedDataLocationsDialog({
  allowed,
  isOpen,
  onClose,
  onSave,
  isPending,
  error,
}: {
  /** The stored list; null accepts any location. */
  allowed: readonly string[] | null
  isOpen: boolean
  onClose: () => void
  onSave: (allowed: string[] | null) => void
  isPending: boolean
  error: unknown
}) {
  const [draft, setDraft] = useState(() => ({
    scope: scopeOf(allowed),
    locations: [...(allowed ?? [])],
  }))
  const { isDirty } = useDirtySnapshot(draft)
  const canSubmit = draft.scope !== "only" || draft.locations.length > 0
  // Held while the save is in flight: an edit made then would be dropped when
  // the save that does not carry it closes the dialog.
  const edit = (next: typeof draft) => {
    if (!isPending) setDraft(next)
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title="Allowed data locations"
      description="Hosted providers serve this workspace only from a region whose declared data location is on the list. A region nobody declared a location for never matches one."
      submitLabel="Save"
      onSubmit={() => {
        if (!canSubmit) return
        onSave(
          draft.scope === "any"
            ? null
            : draft.scope === "none"
              ? []
              : draft.locations,
        )
      }}
      isPending={isPending}
      isSubmitDisabled={!canSubmit}
      isDirty={isDirty}
      error={error}
    >
      <RadioGroup
        label="Data may be kept in"
        value={draft.scope}
        onChange={(value) =>
          edit({
            ...draft,
            scope:
              SCOPE_OPTIONS.find((option) => option.value === value)?.value ??
              "any",
          })
        }
        options={SCOPE_OPTIONS}
        isDisabled={isPending}
      />
      {draft.scope === "only" ? (
        <MultiSelect
          label="Locations"
          description="European Union admits each member state. Global matches a region declared global, not every region."
          options={LOCATION_OPTIONS}
          value={draft.locations}
          onChange={(locations) => edit({ ...draft, locations })}
          searchPlaceholder="Search countries…"
          countNoun={{ one: "location", other: "locations" }}
          isInvalid={draft.locations.length === 0}
          errorMessage={
            draft.locations.length === 0 ? "Pick at least one." : undefined
          }
        />
      ) : null}
    </FormDialog>
  )
}
