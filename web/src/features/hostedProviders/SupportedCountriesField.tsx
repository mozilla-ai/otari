import { Field } from "@/design-system/forms/Field"

import type { ParsedCountries } from "./supportedCountries"

/**
 * The countries whose requests a hosted provider serves. Shared by the add and
 * edit dialogs; the parent parses the text so its save can wait on it.
 */
export function SupportedCountriesField({
  value,
  parsed,
  onChange,
}: {
  value: string
  parsed: ParsedCountries
  onChange: (value: string) => void
}) {
  return (
    <Field
      label="Supported countries"
      value={value}
      onChange={onChange}
      placeholder="Every country"
      description="Two-letter country codes, such as US, DE, GB. A request from any other country, or from one Otari cannot place, is not served here. Needs a country source in the gateway's configuration."
      isInvalid={!parsed.ok}
      errorMessage={
        parsed.ok
          ? undefined
          : `Not a two-letter country code: ${parsed.invalid.join(", ")}`
      }
    />
  )
}
