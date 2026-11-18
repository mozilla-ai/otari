import type { ReactNode } from "react"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"

/**
 * Which of the three the key does about a mismatched `user` field. The wire
 * spells it `true`, `false` and `null`, converted at the two edges below.
 */
export type UserMismatchChoice = "inherit" | "reject" | "accept"

export const userMismatchChoice = (
  stored: boolean | null | undefined,
): UserMismatchChoice =>
  stored === null || stored === undefined
    ? "inherit"
    : stored
      ? "reject"
      : "accept"

export const storedUserMismatch = (
  choice: UserMismatchChoice,
): boolean | null => (choice === "inherit" ? null : choice === "reject")

export function UserMismatchPicker({
  value,
  onChange,
  description,
}: {
  value: UserMismatchChoice
  onChange: (value: UserMismatchChoice) => void
  description: ReactNode
}) {
  const selectId = "key-reject-user-mismatch"
  return (
    <div className="flex flex-col gap-1">
      {/* The label carries markup (`user` as code), which FilterSelect's own
          `label` cannot, so it stays here with `htmlFor` on the trigger. */}
      <label htmlFor={selectId} className="text-body">
        When a request names a different <code>user</code>
      </label>
      <FilterSelect
        id={selectId}
        ariaLabel="When a request names a different user"
        value={value}
        onChange={(next) => onChange(next as UserMismatchChoice)}
        options={[
          { value: "inherit", label: "Use the deployment setting (default)" },
          { value: "reject", label: "Always reject (403)" },
          { value: "accept", label: "Always accept" },
        ]}
      />
      <span className="text-caption">{description}</span>
    </div>
  )
}
