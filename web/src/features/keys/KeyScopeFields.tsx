import type { KeyScope } from "@/client"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Disclosure } from "@/design-system/navigation/Disclosure"
import { GROUPS, SCOPE_INFO, SCOPES } from "@/features/keys/scopes"

const END_USER_SCOPES: ReadonlySet<KeyScope> = new Set(
  GROUPS.filter((group) => group.serviceKeyOnly).flatMap(
    (group) => group.scopes,
  ),
)

/** What a new key holds unless the form says otherwise, as the gateway's default does. */
export const DEFAULT_KEY_SCOPES: readonly KeyScope[] = [
  "inference:run",
  "web:search",
  "web:fetch",
  "code:execute",
  "mcp:read",
  "mcp:execute",
]

/** The scopes once each, in the gateway's order, so two lists compare equal when they hold the same. */
const ordered = (scopes: readonly KeyScope[]): KeyScope[] =>
  SCOPES.map(([scope]) => scope).filter((scope) => scopes.includes(scope))

/**
 * The scopes a key is sent with: a key that is not a service key cannot hold
 * the end-user ones, so they are left off rather than refused.
 */
export const heldScopes = (
  scopes: readonly KeyScope[],
  isServiceKey: boolean,
): KeyScope[] =>
  ordered(scopes).filter((scope) => isServiceKey || !END_USER_SCOPES.has(scope))

type ScopeGroup = (typeof GROUPS)[number]

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
  const advanced = groups.filter((group) => group.isAdvanced)
  const advancedHeld = advanced
    .flatMap((group) => group.scopes)
    .filter((scope) => held.includes(scope)).length

  const set = (scopes: readonly KeyScope[], isSelected: boolean) =>
    onChange(
      ordered(
        isSelected
          ? [...value, ...scopes]
          : value.filter((listed) => !scopes.includes(listed)),
      ),
    )

  const grid = (shown: ScopeGroup[]) => (
    <div className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
      {shown.map((group) => (
        <ScopeGroupFields key={group.key} group={group} held={held} set={set} />
      ))}
    </div>
  )

  return (
    <fieldset className="flex flex-col gap-3">
      <legend className="mb-1 text-body">Access</legend>
      <p className="text-caption">
        What this key may call. Grant only what it needs.
      </p>
      {grid(groups.filter((group) => !group.isAdvanced))}
      {/* Open from the start on a key that already holds one of these, so
          nothing it holds is out of sight. */}
      <Disclosure
        heading={
          <span className="flex items-baseline gap-2">
            <span className="text-emphasis">Advanced scopes</span>
            {advancedHeld > 0 ? (
              <span className="text-caption">{advancedHeld} granted</span>
            ) : null}
          </span>
        }
        isDefaultExpanded={advancedHeld > 0}
        isFlush
      >
        {grid(advanced)}
      </Disclosure>
      {held.length === 0 ? (
        <p role="alert" className="text-caption text-danger">
          Choose at least one
        </p>
      ) : null}
    </fieldset>
  )
}

function ScopeGroupFields({
  group,
  held,
  set,
}: {
  group: ScopeGroup
  held: readonly KeyScope[]
  set: (scopes: readonly KeyScope[], isSelected: boolean) => void
}) {
  const count = group.scopes.filter((scope) => held.includes(scope)).length
  return (
    <div className="flex flex-col">
      {group.scopes.length > 1 ? (
        <Checkbox
          isSelected={count === group.scopes.length}
          isIndeterminate={count > 0 && count < group.scopes.length}
          onChange={(isSelected) => set(group.scopes, isSelected)}
          ariaLabel={`All ${group.label.toLowerCase()} scopes`}
        >
          <span className="text-emphasis">{group.label}</span>
        </Checkbox>
      ) : (
        <span className="text-emphasis">{group.label}</span>
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
                {info.hint ? (
                  <span className="text-caption">{info.hint}</span>
                ) : null}
                {info.risk ? (
                  <span className="text-caption text-warning">{info.risk}</span>
                ) : null}
              </span>
            </Checkbox>
          )
        })}
      </div>
    </div>
  )
}
