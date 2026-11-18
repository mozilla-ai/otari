import { useState } from "react"

import { ComboBoxField } from "@/design-system/forms/ComboBoxField"

import { matchDataLocations } from "./dataLocations"

/** A popover cannot usefully hold every country; typing narrows it. */
const MAX_OPTIONS = 60

/**
 * Where the upstream keeps request data: Global, the European Union, or one
 * country. Empty is undeclared, which a workspace that lists the locations it
 * accepts never matches. Not who may call it: that is "Available to requests
 * from".
 */
export function DataLocationField({
  value,
  onChange,
  description = "Optional. Where this upstream keeps request data. A workspace limited to certain locations refuses a provider with no declared location.",
  placeholder = "Not declared",
}: {
  /** A location code, or empty for undeclared. */
  value: string
  onChange: (value: string) => void
  description?: string
  placeholder?: string
}) {
  const [query, setQuery] = useState("")
  return (
    <ComboBoxField
      label="Data hosted in"
      value={value}
      onChange={onChange}
      onQueryChange={setQuery}
      options={matchDataLocations(query, MAX_OPTIONS, value)}
      placeholder={placeholder}
      shouldSelectOnFocus
      noMatchesMessage="No country or region matches what you typed."
      description={description}
    />
  )
}
