import { useState } from "react"

import type { KeyScope } from "@/client"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Select } from "@/design-system/forms/Select"

type ScopeInfo = {
  scope: KeyScope
  label: string
  /** Set where a leaked key holding the scope could do lasting harm. */
  risk?: string
}

/**
 * Every scope, in the gateway's order (`types/key_scope.py`). A record rather
 * than a list so that a scope the gateway adds fails the typecheck until it is
 * described here: one missing would be dropped from a key on its next save.
 * The end-user scopes come last and are offered only on a service key.
 */
const SCOPE_INFO: Record<KeyScope, Omit<ScopeInfo, "scope">> = {
  inference: { label: "Call models" },
  "files:read": { label: "List and download files" },
  "files:write": {
    label: "Upload and delete files",
    risk: "Can delete files",
  },
  "batches:read": { label: "List batches and read their results" },
  "batches:write": {
    label: "Create and cancel batches",
    risk: "Can queue spend that runs after the key is revoked",
  },
  web: { label: "Search and fetch the web" },
  "code:execute": {
    label: "Run code in the sandbox",
    risk: "Runs arbitrary code",
  },
  "mcp:read": { label: "List MCP servers and their tools" },
  "mcp:execute": {
    label: "Call MCP tools",
    risk: "Acts on the systems those tools reach",
  },
  "telemetry:write": { label: "Send OpenTelemetry data" },
  agents: { label: "Coding-agent hooks and model recommendations" },
  "usage:read": { label: "Read its own budget and usage" },
  "usage:write": { label: "Report usage observed elsewhere" },
  "end_users:read": { label: "List its end users" },
  "end_users:write": { label: "Create, move and block its end users" },
  "end_users:delete": {
    label: "Delete its end users",
    risk: "Erases their records for good",
  },
}

const SCOPES: readonly ScopeInfo[] = Object.entries(SCOPE_INFO).map(
  ([scope, info]) => ({ scope: scope as KeyScope, ...info }),
)

const END_USER_SCOPES: ReadonlySet<KeyScope> = new Set([
  "end_users:read",
  "end_users:write",
  "end_users:delete",
])

/** What a new key holds unless the form says otherwise, as the gateway's default does. */
export const DEFAULT_KEY_SCOPES: readonly KeyScope[] = ["inference"]

type Preset = {
  id: string
  label: string
  description: string
  scopes: readonly KeyScope[]
  serviceKeyOnly?: boolean
}

const PRESETS: readonly Preset[] = [
  {
    id: "models",
    label: "Call models",
    description: "Model calls alone. Enough for an app or an SDK.",
    scopes: DEFAULT_KEY_SCOPES,
  },
  {
    id: "coding-agent",
    label: "Coding agent",
    description:
      "Model calls, plus the Claude Code plugin's hooks, routing and telemetry.",
    scopes: ["inference", "telemetry:write", "agents"],
  },
  {
    id: "usage-import",
    label: "Usage import",
    description:
      "Reports usage and telemetry from elsewhere, and calls no model.",
    scopes: ["telemetry:write", "usage:write"],
  },
  {
    id: "service",
    label: "Service with end users",
    description:
      "Model calls, plus listing and managing its end users. Deleting them is not included.",
    scopes: ["inference", "end_users:read", "end_users:write"],
    serviceKeyOnly: true,
  },
]

const CUSTOM = "custom"

/** The scopes once each, in the gateway's order, so two lists compare equal when they hold the same. */
const ordered = (scopes: readonly KeyScope[]): KeyScope[] =>
  SCOPES.map((info) => info.scope).filter((scope) => scopes.includes(scope))

/**
 * The scopes a key is sent with: a key that is not a service key cannot hold
 * the end-user ones, so they are left off rather than refused.
 */
export const heldScopes = (
  scopes: readonly KeyScope[],
  isServiceKey: boolean,
): KeyScope[] =>
  ordered(scopes).filter((scope) => isServiceKey || !END_USER_SCOPES.has(scope))

const presetFor = (
  scopes: readonly KeyScope[],
  isServiceKey: boolean,
): Preset | undefined => {
  const held = heldScopes(scopes, isServiceKey).join()
  return PRESETS.find(
    (preset) =>
      (isServiceKey || !preset.serviceKeyOnly) &&
      ordered(preset.scopes).join() === held,
  )
}

/**
 * Which endpoint groups the key may call. A preset covers the common uses, and
 * "Custom" lists every scope, so a key can hold the least it needs.
 */
export function KeyScopeFields({
  value,
  onChange,
  isServiceKey,
}: {
  value: readonly KeyScope[]
  onChange: (value: KeyScope[]) => void
  /** Whether the key is a service key, which alone may hold the end-user scopes. */
  isServiceKey: boolean
}) {
  // Custom stays open once chosen, even when the boxes happen to match a preset.
  const [isCustom, setIsCustom] = useState(
    () => presetFor(value, isServiceKey) === undefined,
  )
  const preset = isCustom ? undefined : presetFor(value, isServiceKey)
  const showsList = isCustom || preset === undefined
  const held = heldScopes(value, isServiceKey)
  const offered = SCOPES.filter(
    (info) => isServiceKey || !END_USER_SCOPES.has(info.scope),
  )

  const toggle = (scope: KeyScope, isSelected: boolean) =>
    onChange(
      ordered(
        isSelected
          ? [...value, scope]
          : value.filter((listed) => listed !== scope),
      ),
    )

  return (
    <div className="flex flex-col gap-3 border border-control-border p-3">
      <Select
        label="Access"
        value={preset?.id ?? CUSTOM}
        onChange={(id) => {
          const picked = PRESETS.find((candidate) => candidate.id === id)
          setIsCustom(picked === undefined)
          if (picked) onChange([...picked.scopes])
        }}
        options={[
          ...PRESETS.filter(
            (candidate) => isServiceKey || !candidate.serviceKeyOnly,
          ).map((candidate) => ({
            value: candidate.id,
            label: candidate.label,
          })),
          { value: CUSTOM, label: "Custom" },
        ]}
        description={
          preset
            ? preset.description
            : "What this key may call. A leaked key can do only this, whatever its owner can do."
        }
        isInvalid={held.length === 0}
        errorMessage={held.length === 0 ? "Choose at least one" : undefined}
      />
      {showsList ? (
        <fieldset className="flex flex-col gap-1">
          <legend className="sr-only">Scopes</legend>
          {offered.map((info) => (
            <Checkbox
              key={info.scope}
              isSelected={held.includes(info.scope)}
              onChange={(isSelected) => toggle(info.scope, isSelected)}
              hasTouchTarget
            >
              <span className="flex flex-col">
                <span>
                  <span className="text-foreground">{info.label}</span>{" "}
                  <code className="text-caption">{info.scope}</code>
                </span>
                {info.risk ? (
                  <span className="text-caption text-warning">{info.risk}</span>
                ) : null}
              </span>
            </Checkbox>
          ))}
        </fieldset>
      ) : null}
      <p className="text-caption">No scope lets a key create or manage keys.</p>
    </div>
  )
}
