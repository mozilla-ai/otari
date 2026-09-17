import { Description, Label, TextArea, TextField } from "@heroui/react"

import { FieldMessages } from "@/design-system/forms/FieldMessages"

// The per-call parameters this gateway has no field for.
//
// Not an advanced-user affordance but the thing that keeps the form from being
// narrower than the API behind it: the catalog describes the any-guardrail this
// gateway ships, and a guardrail may take an argument a newer one added. Sent
// under whatever the typed fields set, so a key in both is the field's.

export function GuardrailExtraJsonField({
  value,
  error,
  disabled,
  onChange,
}: {
  value: string
  error: string | undefined
  disabled?: boolean
  onChange: (next: string) => void
}) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      isDisabled={disabled}
      isInvalid={Boolean(error)}
      className="flex flex-col gap-1"
    >
      <Label className="text-body">Other per-call parameters</Label>
      <TextArea
        rows={3}
        spellCheck={false}
        placeholder={'{"threshold": 0.8}'}
        className="font-mono text-xs"
      />
      <FieldMessages>
        <Description
          className={error ? "text-caption text-danger" : "text-caption"}
        >
          {error ??
            "Sent with the text on every check, under whatever the fields above set. For a parameter this gateway has no field for."}
        </Description>
      </FieldMessages>
    </TextField>
  )
}
