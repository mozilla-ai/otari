import { useState } from "react"
import { FileTrigger } from "react-aria-components"
import { FiUpload } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { Field } from "@/design-system/forms/Field"

import {
  countriesFromJson,
  formatCountries,
  type ParsedCountries,
} from "./supportedCountries"

/**
 * The countries whose requests a hosted provider serves. Shared by the add and
 * edit dialogs; the parent parses the text so its save can wait on it. A long
 * list can be loaded from a JSON file, which fills the field to be reviewed.
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
  const [fileError, setFileError] = useState<string | null>(null)

  const load = async (file: File) => {
    const loaded = countriesFromJson(await file.text())
    if (loaded.ok) {
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
        label="Supported countries"
        value={value}
        onChange={(next) => {
          setFileError(null)
          onChange(next)
        }}
        placeholder="Every country"
        description="Two-letter country codes, such as US, DE, GB, typed or loaded from a JSON list. A request from any other country, or from one Otari cannot place, is not served here. Needs a country source: a header or a country file in the gateway's configuration, or a country file stored on the Settings page."
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
