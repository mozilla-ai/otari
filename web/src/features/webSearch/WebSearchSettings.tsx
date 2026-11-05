import { useState } from "react"

import type { ToolSettingField } from "@/client"
import { Toggle } from "@/design-system/forms/Toggle"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { AdvancedRows } from "@/features/tools/AdvancedRows"
import {
  copyFor,
  oneField,
  type SettingCopy,
  ToolSettingRow,
} from "@/features/tools/ToolSettingRows"
import { useUpdateToolSettings } from "@/shared/api/tools"
import { useAutosave } from "@/shared/hooks/useAutosave"

import { isNoSearch } from "./webTools"

// The deployment-wide web search settings: which tool each of the model's web
// tools uses, the per-request cap, how results come back, and the legacy
// SearXNG backend while it exists. Each row saves itself, as on the Tools page.

const COPY: Readonly<Record<string, SettingCopy>> = {
  web_search_max_calls: {
    label: "Calls per request",
    help: "How many searches and fetches one request may make together, on any tool. A call past it gets a tool error the model can recover from.",
    placeholder: "10",
  },
  web_search_max_results: {
    label: "Results per search",
    help: "How many hits a search returns when the request names no max_results, on any tool. A workspace's own limit can only lower it.",
    placeholder: "10",
  },
  web_search_extract: {
    label: "Read pages for hits without text",
    help: "On: a hit that comes back with no page text is read through the search tool's fetch tool. Off: hits keep only their snippets.",
    placeholder: "",
    defaultLabel: "Default (on)",
  },
  web_search_intercept: {
    label: "Run provider web search here",
    help: "When a request asks for the model provider's own search (OpenAI's web_search, Anthropic's web_search_20250305), run it on these search tools instead, at these prices and within the workspace's domain lists. Off: the provider runs it and bills for it.",
    placeholder: "",
    defaultLabel: "Default (off)",
  },
  web_search_url: {
    label: "Backend URL",
    help: "A SearXNG-compatible service the model searches when no search tool is chosen for it. A searxng search tool with no endpoint of its own uses it too.",
    placeholder: "http://searxng:8080",
    isMachineReadable: true,
  },
  web_search_engines: {
    label: "Engines",
    help: "Comma-separated SearXNG engines. Blank uses the backend's defaults.",
    placeholder: "google,bing,duckduckgo",
    isMachineReadable: true,
  },
}

/** Set from the tables' rows, or by the switch below. */
const DEFAULT_KEYS = ["web_search_default_tool", "web_fetch_default_tool"]
const BEHAVIOR_KEYS = [
  "web_search_max_calls",
  "web_search_max_results",
  "web_search_extract",
  "web_search_intercept",
]
const LEGACY_KEYS = ["web_search_url", "web_search_engines"]
/** Set through the API only. */
const OMITTED = ["web_search_purpose_hint"]

const isSet = (field: ToolSettingField | undefined) =>
  field !== undefined && field.value !== null && field.value !== ""

/**
 * Whether the model may search on this deployment's tools: `none` turns it
 * off, and turning it back on clears the setting, which leaves the default to
 * the stars, the configuration file, or automatic. A configuration file that
 * says `none` itself keeps search off, so the row says how to turn it on.
 */
function ModelSearchRow({
  field,
  disabled,
  commit,
}: {
  field: ToolSettingField
  disabled: boolean
  commit: (value: string | null) => Promise<unknown>
}) {
  const save = useAutosave()
  const [wasTurnedOn, setWasTurnedOn] = useState(false)
  const isOff = typeof field.value === "string" && isNoSearch(field.value)
  return (
    <SettingRow
      label="The model can search"
      configKey={field.key}
      help="Off: the model searches only on an organization's own key, never on these tools. On: it uses the starred search tool, else the only one, or the legacy backend while it is set."
      error={save.error}
      note={
        wasTurnedOn && isOff && !save.isSaving ? (
          <p className="text-caption">
            The configuration file sets none, so search stays off: star a search
            tool to turn it on.
          </p>
        ) : null
      }
      control={
        <Toggle
          label="The model can search"
          isSelected={!isOff}
          isDisabled={disabled || save.isSaving}
          onChange={(next) => {
            setWasTurnedOn(next)
            void save.run(() => commit(next ? null : "none"))
          }}
        />
      }
    />
  )
}

export function WebSearchSettings({
  fields,
  disabled,
}: {
  /** The tool settings' fields, every service's. */
  fields: readonly ToolSettingField[]
  disabled: boolean
}) {
  const update = useUpdateToolSettings()
  const own = fields.filter((field) => field.service === "web_search")
  const byKey = new Map(own.map((field) => [field.key, field]))
  const known = new Set([
    ...DEFAULT_KEYS,
    ...BEHAVIOR_KEYS,
    ...LEGACY_KEYS,
    ...OMITTED,
  ])
  // A key the backend adds still renders, so it is visible without a change here.
  const unlisted = own.filter((field) => !known.has(field.key))
  const fieldsFor = (keys: readonly string[]) =>
    keys.flatMap((key) => {
      const field = byKey.get(key)
      return field ? [field] : []
    })

  const renderField = (field: ToolSettingField) => {
    const copy = copyFor(COPY, field)
    return (
      <ToolSettingRow
        key={field.key}
        field={field}
        copy={copy}
        defaultLabel={copy.defaultLabel}
        commit={(key, value) => update.mutateAsync(oneField(key, value))}
        disabled={disabled}
        readOnly={false}
      />
    )
  }

  const legacy = fieldsFor(LEGACY_KEYS)
  const searchDefault = byKey.get("web_search_default_tool")

  return (
    <>
      <SettingsGroup
        title="The model's web tools"
        description="How the model searches and reads pages, on whichever tool serves it. The search API, POST /api/v1/search, is not affected. The star on a tool's row makes it the default."
      >
        {searchDefault ? (
          <ModelSearchRow
            field={searchDefault}
            disabled={disabled}
            commit={(value) =>
              update.mutateAsync(oneField(searchDefault.key, value))
            }
          />
        ) : null}
        {fieldsFor(BEHAVIOR_KEYS).map(renderField)}
        {unlisted.length > 0 ? (
          <AdvancedRows>{unlisted.map(renderField)}</AdvancedRows>
        ) : null}
      </SettingsGroup>
      {legacy.some(isSet) ? (
        <SettingsGroup
          title="Legacy SearXNG backend"
          description="The model's search backend from before search tools. Keep it until a searxng search tool can serve the model."
        >
          {legacy.map(renderField)}
        </SettingsGroup>
      ) : null}
    </>
  )
}
