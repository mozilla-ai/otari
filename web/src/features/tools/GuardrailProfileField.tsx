import { Button } from "@heroui/react"
import { useState } from "react"

import type { GuardrailCatalog } from "@/client"
import { findProfile } from "@/features/tools/guardrailParameters"
import { Field } from "@/shared/components/forms/Field"
import { FieldMessages } from "@/shared/components/forms/FieldMessages"
import { FilterSelect } from "@/shared/components/navigation/FilterSelect"

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
  value,
  disabled,
  onChange,
}: {
  catalog: GuardrailCatalog | undefined
  value: string
  disabled: boolean
  onChange: (next: string) => void
}) {
  const profiles = catalog?.profiles ?? []
  const listed = catalog?.available === true && profiles.length > 0
  const [byHand, setByHand] = useState(false)
  const chosen = findProfile(catalog, value)

  if (!listed || byHand) {
    return (
      <div className="flex flex-col gap-1">
        <Field
          label="Guardrail profile"
          value={value}
          onChange={onChange}
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
      <FilterSelect
        label="Guardrail profile"
        value={chosen ? value : ""}
        disabled={disabled}
        onChange={onChange}
        options={[
          { value: "", label: "Choose a profile" },
          ...profiles.map((profile) => ({
            value: profile.profile,
            label: profile.profile,
          })),
        ]}
      />
      <FieldMessages reserve>
        <span className="text-caption">
          {chosen
            ? `Runs ${chosen.guardrail}${chosen.model_id ? ` on ${chosen.model_id}` : ""} on the guardrails service.`
            : "Built by this deployment's guardrails service from the operator's own configuration."}
        </span>
      </FieldMessages>
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
