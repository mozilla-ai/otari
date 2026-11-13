import type { KeyScope } from "@/client"
import { Checkbox } from "@/design-system/forms/Checkbox"

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

type ScopeGroup = {
  label: string
  scopes: readonly KeyScope[]
  serviceKeyOnly?: boolean
}

/** How the form lays the scopes out. Each scope belongs to exactly one group. */
const GROUPS: readonly ScopeGroup[] = [
  { label: "Models", scopes: ["inference"] },
  { label: "Files", scopes: ["files:read", "files:write"] },
  { label: "Batches", scopes: ["batches:read", "batches:write"] },
  {
    label: "Tools",
    scopes: ["web", "code:execute", "mcp:read", "mcp:execute"],
  },
  { label: "Coding agents", scopes: ["agents", "telemetry:write"] },
  { label: "Usage", scopes: ["usage:read", "usage:write"] },
  {
    label: "End users",
    scopes: ["end_users:read", "end_users:write", "end_users:delete"],
    serviceKeyOnly: true,
  },
]

const END_USER_SCOPES: ReadonlySet<KeyScope> = new Set(
  GROUPS.filter((group) => group.serviceKeyOnly).flatMap(
    (group) => group.scopes,
  ),
)

/** What a new key holds unless the form says otherwise, as the gateway's default does. */
export const DEFAULT_KEY_SCOPES: readonly KeyScope[] = ["inference"]

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

/** Which endpoint groups the key may call. */
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
  const held = heldScopes(value, isServiceKey)
  const groups = GROUPS.filter((group) => isServiceKey || !group.serviceKeyOnly)

  const set = (scopes: readonly KeyScope[], isSelected: boolean) =>
    onChange(
      ordered(
        isSelected
          ? [...value, ...scopes]
          : value.filter((listed) => !scopes.includes(listed)),
      ),
    )

  return (
    <fieldset className="flex flex-col gap-3 border border-control-border p-3">
      <legend className="px-1 text-sm font-medium text-foreground">
        Access
      </legend>
      <p className="text-caption">
        What this key may call. Grant only what it needs.
      </p>
      <div className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
        {groups.map((group) => {
          const count = group.scopes.filter((scope) =>
            held.includes(scope),
          ).length
          return (
            <div key={group.label} className="flex flex-col">
              {group.scopes.length > 1 ? (
                <Checkbox
                  isSelected={count === group.scopes.length}
                  isIndeterminate={count > 0 && count < group.scopes.length}
                  onChange={(isSelected) => set(group.scopes, isSelected)}
                  ariaLabel={`All ${group.label.toLowerCase()} scopes`}
                >
                  <span className="font-medium text-foreground">
                    {group.label}
                  </span>
                </Checkbox>
              ) : (
                <span className="font-medium text-body text-foreground">
                  {group.label}
                </span>
              )}
              <div
                className={`mt-1 flex flex-col gap-1.5${group.scopes.length > 1 ? " pl-6" : ""}`}
              >
                {group.scopes.map((scope) => {
                  const info = SCOPE_INFO[scope]
                  return (
                    <Checkbox
                      key={scope}
                      isSelected={held.includes(scope)}
                      onChange={(isSelected) => set([scope], isSelected)}
                    >
                      <span className="flex flex-col">
                        <span className="text-foreground">{info.label}</span>
                        <code className="text-caption">{scope}</code>
                        {info.risk ? (
                          <span className="text-caption text-warning">
                            {info.risk}
                          </span>
                        ) : null}
                      </span>
                    </Checkbox>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
      {held.length === 0 ? (
        <p role="alert" className="text-caption text-danger">
          Choose at least one
        </p>
      ) : null}
    </fieldset>
  )
}
