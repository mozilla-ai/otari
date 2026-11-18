import { useState } from "react"
import { FileTrigger } from "react-aria-components"
import { FiUpload } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { Field } from "@/design-system/forms/Field"

import {
  countriesFromJson,
  formatCountries,
  isCountryCode,
  type ParsedCountries,
} from "./supportedCountries"

/**
 * The countries whose requests a hosted provider serves, which is who may call
 * it rather than where it keeps data ("Data hosted in"). Shared by the add and
 * edit dialogs; the parent parses the text so its save can wait on it. A long
 * list can be loaded from a JSON file, which fills the field to be reviewed.
 */
export function SupportedCountriesField({
  value,
  parsed,
  onChange,
  placeholder = "Every country",
}: {
  value: string
  parsed: ParsedCountries
  onChange: (value: string) => void
  /** What blank means here: every country, or the provider's list for a region. */
  placeholder?: string
}) {
  const [fileError, setFileError] = useState<string | null>(null)

  const load = async (file: File) => {
    const loaded = countriesFromJson(await file.text())
    if (loaded.ok && loaded.value.length === 0) {
      // Blank means "no list" here, while an empty list admits no country,
      // so a file of none is refused rather than read as no restriction.
      setFileError(
        "The file lists no country. To refuse every country, set an empty list through the API.",
      )
    } else if (loaded.ok && !loaded.value.every(isCountryCode)) {
      // Checked here as well as once in the field: an entry such as "" would
      // vanish from the text and leave it reading as every country.
      const invalid = loaded.value.filter((code) => !isCountryCode(code))
      setFileError(
        `The file holds entries that are not two-letter country codes: ${invalid.map((code) => JSON.stringify(code)).join(", ")}`,
      )
    } else if (loaded.ok) {
      setFileError(null)
      onChange(formatCountries(loaded.value))
    } else {
      setFileError(loaded.error)
    }
  }

  const error =
    fileError ??
    (parsed.ok
      ? undefined
      : `Not a two-letter country code: ${parsed.invalid.join(", ")}`)

  return (
    <div className="flex flex-col gap-1">
      <Field
        label="Available to requests from"
        value={value}
        onChange={(next) => {
          setFileError(null)
          onChange(next)
        }}
        placeholder={placeholder}
        description="Who may send requests here, not where data is kept. Two-letter country codes, such as US, DE, GB, typed or loaded from a JSON list. A request from any other country, or from one Otari cannot place, is not served here. Needs a country source: a header or a country file in the gateway's configuration, or a country file stored on the Settings page."
        isInvalid={error !== undefined}
        errorMessage={error}
      />
      <FileTrigger
        acceptedFileTypes={["application/json", ".json"]}
        onSelect={(files) => {
          const file = files?.[0]
          if (file) void load(file)
        }}
      >
        <Button size="sm" className="self-start">
          <FiUpload aria-hidden className="size-4" />
          Load from JSON file
        </Button>
      </FileTrigger>
    </div>
  )
}
