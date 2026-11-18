import { Field } from "@/design-system/forms/Field"
import { Disclosure } from "@/design-system/navigation/Disclosure"

/** A whole number of zero or more, null for blank (no cap). */
export function parseCount(raw: string): {
  value: number | null
  isValid: boolean
} {
  const trimmed = raw.trim()
  if (trimmed === "") return { value: null, isValid: true }
  // Digits first: `Number` rounds "1.0000000000000001" to a whole 1.
  if (!/^\d+$/.test(trimmed)) return { value: null, isValid: false }
  const n = Number(trimmed)
  if (!Number.isSafeInteger(n)) return { value: null, isValid: false }
  return { value: n, isValid: true }
}

export function countToInput(value: number | null | undefined): string {
  return value === null || value === undefined ? "" : String(value)
}

/**
 * The token and request caps a budget holds beside its dollar figure.
 *
 * Behind a disclosure because most budgets cap dollars alone, and open from the
 * start when the budget already caps either, so an edit never hides a cap that
 * is in force.
 */
export function CountLimitFields({
  tokens,
  onTokensChange,
  requests,
  onRequestsChange,
}: {
  tokens: string
  onTokensChange: (value: string) => void
  requests: string
  onRequestsChange: (value: string) => void
}) {
  // Read on mount only, so clearing both while it is open does not snap it shut.
  return (
    <Disclosure
      heading="Token and request limits"
      isDefaultExpanded={tokens !== "" || requests !== ""}
    >
      <div className="flex flex-col gap-3">
        <Field
          label="Token limit"
          value={tokens}
          onChange={onTokensChange}
          placeholder="1000000"
          isInvalid={!parseCount(tokens).isValid}
          errorMessage="Enter a whole number of zero or more, or leave it blank for no limit."
          description="Total tokens per period. Leave blank for no limit."
        />
        <Field
          label="Request limit"
          value={requests}
          onChange={onRequestsChange}
          placeholder="1000"
          isInvalid={!parseCount(requests).isValid}
          errorMessage="Enter a whole number of zero or more, or leave it blank for no limit."
          description="Requests per period. Leave blank for no limit."
        />
      </div>
    </Disclosure>
  )
}
