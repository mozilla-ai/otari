import { Checkbox } from "@/design-system/forms/Checkbox"
import { MultiSelect } from "@/design-system/forms/MultiSelect"
import { DismissChip } from "@/design-system/indicators/DismissChip"

import type { EntityGroup } from "./appliedEntities"

// Where a budget applies, as one set of entity keys split across the groups an
// admin scans: the whole organization, then one picker per kind of entity.
//
// The set is the source of truth and each control edits only its own slice of
// it, so a key no control offers (a ceiling narrowed in a way the groups do not
// express, or an option whose list failed to load) is never dropped by an edit
// elsewhere. Those are listed under "Also applied to", where they can be removed
// on purpose.

export function AppliedToPicker({
  value,
  onChange,
  groups,
  organizationKey,
  organizationName,
  organizationTaken,
  describe,
  isLoading = false,
}: {
  value: readonly string[]
  onChange: (next: string[]) => void
  groups: readonly EntityGroup[]
  /** The entity key of the whole organization, which is a checkbox rather than a list. */
  organizationKey: string
  organizationName: string
  /** What already carries the whole organization, when something other than this budget does. */
  organizationTaken?: string
  /** A plain name for a key no group offers. */
  describe: (key: string) => string
  /** The lists are still loading, so a key no group offers yet is not one no group will. */
  isLoading?: boolean
}) {
  const offered = new Set([
    organizationKey,
    ...groups.flatMap((group) => group.options.map((option) => option.id)),
  ])
  const others = isLoading ? [] : value.filter((key) => !offered.has(key))
  const wholeOrganization = value.includes(organizationKey)

  const replaceSlice = (slice: ReadonlySet<string>, next: readonly string[]) =>
    onChange([...value.filter((key) => !slice.has(key)), ...next])

  return (
    <fieldset className="flex flex-col gap-4">
      <legend className="text-body pb-1">Applied to</legend>
      <p className="text-sm text-muted">
        Each entity draws on its own allowance of the limit. An entity carries
        one budget, so one already on another budget is listed but cannot be
        picked.
      </p>
      <div className="flex flex-col gap-0.5">
        <Checkbox
          isSelected={wholeOrganization}
          // Unticking stays possible when something else carries it, for the
          // reason a disabled option's chip still removes it.
          isDisabled={organizationTaken !== undefined && !wholeOrganization}
          onChange={(next) =>
            replaceSlice(
              new Set([organizationKey]),
              next ? [organizationKey] : [],
            )
          }
        >
          {organizationName} (whole organization)
          {/* The reason it is disabled, as part of what is announced; the
              caption below shows it. */}
          {organizationTaken ? (
            <span className="sr-only">, {organizationTaken}</span>
          ) : null}
        </Checkbox>
        {organizationTaken ? (
          <span className="text-caption text-subtle pl-6">
            {organizationTaken}
          </span>
        ) : null}
      </div>
      <div className="grid gap-x-6 gap-y-4 md:grid-cols-2">
        {groups.map((group) => {
          const slice = new Set(group.options.map((option) => option.id))
          return (
            <MultiSelect
              key={group.id}
              label={group.label}
              options={group.options}
              value={value.filter((key) => slice.has(key))}
              onChange={(next) => replaceSlice(slice, next)}
              // The count noun rather than the label, which would lowercase "API".
              searchPlaceholder={`Search ${group.countNoun.other}…`}
              countNoun={group.countNoun}
              // The ids are entity keys, not anything an admin types.
              searchesId={false}
              emptyMessage={`No ${group.countNoun.other} to choose from.`}
            />
          )
        })}
      </div>
      {others.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-body">Also applied to</span>
          <ul
            aria-label="Also applied to"
            className="flex list-none flex-wrap gap-x-2 gap-y-1.5"
          >
            {others.map((key) => (
              <li key={key}>
                <DismissChip
                  value={describe(key)}
                  onDismiss={() => replaceSlice(new Set([key]), [])}
                  dismissLabel={`Remove ${describe(key)}`}
                />
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </fieldset>
  )
}
