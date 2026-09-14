import { Button } from "@heroui/react"
import { useState } from "react"

import type { GuardrailCatalog } from "@/client"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { findProfile } from "@/features/tools/guardrailParameters"

// Which profile a new entry mandates.
//
// A profile now comes from either of two places, and the picker offers both: a
// key in the guardrails service's own configuration, or a guardrail defined in
// this gateway and run in its process. The second is why a deployment with no
// sidecar at all still gets a list rather than a bare text box.
//
// Typing one by hand stays available, and not only as a fallback for a catalog
// that failed to load: an entry may name an endpoint of its own, whose profiles
// this deployment's service was never asked about.
//
// The two are separate controls rather than a picker with an "other" row in it,
// which would need a sentinel value no profile could ever be called.

// Trails a locally defined name in the picker. `Select` carries no second line
// per row, so the one thing that tells the two sources apart has to ride on the
// label; it is stripped back off before the value is reported.
const LOCAL_SUFFIX = " (defined here)"

export function GuardrailProfileField({
  catalog,
  pending,
  value,
  localNames,
  disabled,
  onChange,
}: {
  catalog: GuardrailCatalog | undefined
  /**
   * The catalog read has not settled yet. Kept distinct from an empty catalog
   * so the control does not start as a text box and turn into a picker under
   * the operator's cursor a moment later.
   */
  pending: boolean
  value: string
  /**
   * Guardrails defined in this gateway, whose names a profile may also be. Empty
   * for a caller who cannot read them, which is every member who is not the
   * deployment's operator; they keep the by-hand box.
   */
  localNames: readonly string[]
  disabled?: boolean
  onChange: (next: string) => void
}) {
  const profiles = catalog?.profiles ?? []
  const serviceListed = catalog?.available === true && profiles.length > 0
  const listed = serviceListed || localNames.length > 0
  const [byHand, setByHand] = useState(false)
  const chosen = findProfile(catalog, value)
  const isLocal = localNames.includes(value)

  if (pending) {
    return (
      <Select
        label="Guardrail profile"
        value=""
        isDisabled
        isRequired
        onChange={onChange}
        options={[]}
        // The waiting sentence is the description rather than the placeholder:
        // HeroUI's trigger renders the selected option's text and treats an
        // empty value as a selection of "", so a placeholder never reaches the
        // trigger here. The description is rendered and announced either way.
        description="Reading the guardrails service…"
      />
    )
  }

  if (!listed || byHand) {
    return (
      <div className="flex flex-col gap-1">
        <Field
          label="Guardrail profile"
          value={value}
          onChange={onChange}
          isRequired
          isDisabled={disabled}
          placeholder="prompt-injection"
          description={
            listed
              ? "A profile on whichever guardrails service this entry is sent to."
              : (catalog?.reason ??
                "The profile has to exist on the guardrails service.")
          }
          reserveMessage
        />
        {listed ? (
          <Button
            size="sm"
            variant="ghost"
            isDisabled={disabled}
            onPress={() => {
              setByHand(false)
              onChange("")
            }}
          >
            Choose a listed profile instead
          </Button>
        ) : null}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      <Select
        label="Guardrail profile"
        value={chosen || isLocal ? value : ""}
        isDisabled={disabled}
        isRequired
        onChange={onChange}
        placeholder="Choose a profile"
        options={[
          // Local first: a gateway that defines its own guardrails is naming
          // the thing it just configured, and the sidecar's list is the older
          // and usually longer one.
          ...localNames.map((name) => ({
            value: name,
            label: `${name}${LOCAL_SUFFIX}`,
          })),
          ...profiles.map((profile) => ({
            value: profile.profile,
            label: profile.profile,
          })),
        ]}
        description={
          isLocal
            ? "Runs in this gateway, from the guardrail defined here. No guardrails service is involved."
            : chosen
              ? `Runs ${chosen.guardrail}${chosen.model_id ? ` on ${chosen.model_id}` : ""} on the guardrails service.`
              : "Defined in this gateway, or built by the guardrails service from the operator's own configuration."
        }
      />
      <Button
        size="sm"
        variant="ghost"
        isDisabled={disabled}
        onPress={() => {
          setByHand(true)
          onChange("")
        }}
      >
        Name a profile by hand
      </Button>
    </div>
  )
}
