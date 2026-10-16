import { Field } from "@/design-system/forms/Field"

import { policyNameProblem } from "./policyModel"

/**
 * Why a name cannot be saved, or "". Only a name being chosen is held to the
 * rules: one the server already stored keeps working as callers send it.
 */
export function nameProblemOf(name: string, previousName = ""): string {
  if (name.trim() === "") return ""
  return previousName !== "" && name.trim() === previousName
    ? ""
    : policyNameProblem(name)
}

/** A policy's name, which is what callers send as `model`. */
export function PolicyNameField({
  value,
  onChange,
  previousName = "",
  autoFocus,
}: {
  value: string
  onChange: (next: string) => void
  /** The stored name, on an edit; a different one renames the policy. */
  previousName?: string
  autoFocus?: boolean
}) {
  const problem = nameProblemOf(value, previousName)
  const isRenaming =
    previousName !== "" && value.trim() !== "" && value.trim() !== previousName
  return (
    <Field
      label="Name"
      value={value}
      onChange={onChange}
      placeholder="fast"
      isRequired
      autoFocus={autoFocus}
      isInvalid={problem !== ""}
      errorMessage={problem}
      description={
        isRenaming ? (
          <span>
            Renames <code>{previousName}</code> on save. Callers have to send
            the new name from then on, and usage already recorded keeps the old
            one.
          </span>
        ) : value.trim() !== "" ? (
          <>
            Callers send <code>model: "{value.trim()}"</code>.
          </>
        ) : (
          <>
            What callers send as <code>model</code>. No spaces, “:” or “/”, and
            not the name of a provider or an alias.
          </>
        )
      }
    />
  )
}
