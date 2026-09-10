import { Button, Description, Label, TextArea, TextField } from "@heroui/react"
import { useState } from "react"
import { FiChevronDown } from "react-icons/fi"

import type { GuardrailParameterSpec } from "@/client"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { GuardrailParameterFields } from "@/features/tools/GuardrailParameterFields"
import type {
  ParameterErrors,
  ParameterValues,
} from "@/features/tools/guardrailParameters"

// The `validate_kwargs` half of a guardrail entry: the fields the chosen
// profile's schema describes, plus the raw editor for everything it does not.
//
// Both halves are always available, and that is the design rather than an
// oversight. The catalog describes the guardrail classes this gateway's
// any-guardrail knows; the guardrails service may run a newer one, may be
// unreachable while the operator is configuring an entry, and an entry may point
// at an endpoint of its own that this catalog never described. In each of those
// the typed fields are absent and the entry still has to be configurable, so the
// escape hatch is not an advanced-user affordance: it is what keeps the form
// from being narrower than the API behind it.

export function GuardrailParametersSection({
  specs,
  scopeName,
  values,
  errors,
  extraJson,
  extraJsonError,
  described,
  disabled,
  onChange,
  onExtraJsonChange,
}: {
  specs: GuardrailParameterSpec[]
  /** Names the entry, so repeated parameter labels stay distinguishable. */
  scopeName: string
  values: ParameterValues
  errors: ParameterErrors
  extraJson: string
  extraJsonError: string | undefined
  /**
   * Whether the catalog has an entry for this profile. False when the service
   * could not be listed, when the entry names an endpoint of its own, or when
   * the profile is one this gateway's any-guardrail has no schema for. A
   * profile with an entry and no parameters is still described: it genuinely
   * takes none.
   */
  described: boolean
  disabled: boolean
  onChange: (name: string, next: ParameterValues[string]) => void
  onExtraJsonChange: (next: string) => void
}) {
  // Open when there is something in it to see, or when it is the only place
  // configuration could go. A raw value the operator cannot find is worse than
  // one more expanded panel; a profile that is described and takes nothing has
  // an empty panel worth collapsing.
  const [open, setOpen] = useState(extraJson !== "" || !described)

  return (
    <div className="flex flex-col gap-3">
      {specs.length > 0 ? (
        <GuardrailParameterFields
          specs={specs}
          scopeName={scopeName}
          values={values}
          errors={errors}
          disabled={disabled}
          onChange={onChange}
        />
      ) : null}
      <div>
        <Button
          size="sm"
          variant="ghost"
          aria-expanded={open}
          onPress={() => setOpen(!open)}
        >
          {`Other parameters for ${scopeName}`}
          <FiChevronDown
            aria-hidden="true"
            className={`transition-transform ${open ? "rotate-180" : ""}`}
          />
        </Button>
      </div>
      {open ? (
        <TextField
          value={extraJson}
          onChange={onExtraJsonChange}
          isDisabled={disabled}
          isInvalid={Boolean(extraJsonError)}
          className="flex max-w-md flex-col gap-1"
        >
          <Label className="text-body">Parameters (JSON)</Label>
          <TextArea
            rows={3}
            spellCheck={false}
            placeholder={'{"threshold": 0.8}'}
            className="font-mono text-xs"
          />
          <FieldMessages>
            <Description
              className={
                extraJsonError ? "text-caption text-danger" : "text-caption"
              }
            >
              {extraJsonError ??
                "Sent to the guardrails service as validate_kwargs, under whatever the fields above set. Use it for a parameter this gateway has no schema for."}
            </Description>
          </FieldMessages>
        </TextField>
      ) : null}
    </div>
  )
}
