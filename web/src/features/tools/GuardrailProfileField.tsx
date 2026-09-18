import { Button } from "@heroui/react"
import { useState } from "react"

import type { GuardrailCatalog } from "@/client"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { findProfile } from "@/features/tools/guardrailParameters"

// Which profile a new entry mandates.
//
// A profile is a key in the operator's own guardrails-service configuration, so
// the list is read from that service rather than kept here. Typing one by hand
// stays available, and not only as a fallback for a catalog that failed to load:
// an entry may name an endpoint of its own, whose profiles this deployment's
// service was never asked about.
//
// The two are separate controls rather than a picker with an "other" row in it,
// which would need a sentinel value no profile could ever be called.

export function GuardrailProfileField({
  catalog,
  isPending,
  value,
  disabled,
  onChange,
}: {
  catalog: GuardrailCatalog | undefined
  /**
   * The catalog read has not settled yet. Kept distinct from an empty catalog
   * so the control does not start as a text box and turn into a picker under
   * the operator's cursor a moment later.
   */
  isPending: boolean
  value: string
  disabled?: boolean
  onChange: (next: string) => void
}) {
  const profiles = catalog?.profiles ?? []
  const isListed = catalog?.available === true && profiles.length > 0
  const [byHand, setByHand] = useState(false)
  const chosen = findProfile(catalog, value)

  if (isPending) {
    return (
      <Select
        label="Guardrail profile"
        value=""
        isDisabled
        isRequired
        onChange={onChange}
        options={[]}
        // The waiting sentence is the description and not the placeholder: a
        // description is announced and a trigger's value slot is not, so this
        // way a screen reader is told the field is still loading.
        description="Reading the guardrails service…"
      />
    )
  }

  if (!isListed || byHand) {
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
            isListed
              ? "A profile on whichever guardrails service this entry is sent to."
              : (catalog?.reason ??
                "The profile has to exist on the guardrails service.")
          }
          shouldReserveMessage
        />
        {isListed ? (
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
        value={chosen ? value : ""}
        isDisabled={disabled}
        isRequired
        onChange={onChange}
        placeholder="Choose a profile"
        options={profiles.map((profile) => ({
          value: profile.profile,
          label: profile.profile,
        }))}
        description={
          chosen
            ? `Runs ${chosen.guardrail}${chosen.model_id ? ` on ${chosen.model_id}` : ""} on the guardrails service.`
            : "Built by this deployment's guardrails service from the operator's own configuration."
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
