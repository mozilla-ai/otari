import { useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { SearchField } from "@/design-system/forms/SearchField"
import { Dot } from "@/design-system/indicators/Dot"
import { isToolPattern, toolsMatching } from "@/features/tools/mcpAccess"

function toolNoun(count: number) {
  return count === 1 ? "tool" : "tools"
}

/**
 * Picks tool names for a server: what goes on its allow-list, which tools are
 * labeled read, or what an access rule allows. One component for all three, so
 * each reads the same.
 *
 * Once a check has listed the server's tools (`isKnown`), they are the list:
 * each with the first line of its own description, a search over both, and
 * Select all / Select none over what the search shows. A pattern such as
 * `create_*` can still be typed, and says how many of the known tools it
 * matches; an entry that matches none of them is flagged, since it would allow
 * nothing today. Before any check there is nothing to pick from, so names are
 * typed, and the Check action sits right there.
 *
 * Controlled: `selected` is the entries in order, and every change hands back
 * the whole new list. Stored values are exactly what was picked or typed.
 */
export function ToolChecklist({
  label,
  tools,
  descriptions = {},
  isKnown,
  selected,
  onChange,
  allowPatterns = true,
  isDisabled = false,
  onCheck,
  isChecking = false,
}: {
  /** Names the list for assistive technology; the page's own heading shows it. */
  label: string
  /** The server's tools, as its last passing check listed them. */
  tools: readonly string[]
  /** The first line of each tool's own description, where it has one. */
  descriptions?: Readonly<Record<string, string>>
  /** Whether a passing check listed the tools; false before any check. */
  isKnown: boolean
  selected: readonly string[]
  onChange: (next: string[]) => void
  /** A rule takes `prefix*` patterns; the server's own allow-list and read labels take names only. */
  allowPatterns?: boolean
  isDisabled?: boolean
  /** Runs the server's check, offered while the tools are not known. */
  onCheck?: () => void
  isChecking?: boolean
}) {
  const [typed, setTyped] = useState("")
  const [query, setQuery] = useState("")
  const entry = typed.trim()
  const known = isKnown ? tools : []
  // A tool can be named `constructor`, so read only the map's own entries.
  const descriptionOf = (name: string) =>
    Object.hasOwn(descriptions, name) ? descriptions[name] : ""
  const needle = query.trim().toLowerCase()
  const visible = known.filter(
    (name) =>
      needle === "" ||
      name.toLowerCase().includes(needle) ||
      descriptionOf(name).toLowerCase().includes(needle),
  )
  const others = selected.filter((name) => !known.includes(name))
  const pickedKnown = known.filter((name) => selected.includes(name))

  const toggle = (name: string, isSelected: boolean) =>
    onChange(
      isSelected
        ? [...selected, name]
        : selected.filter((existing) => existing !== name),
    )
  const selectVisible = () =>
    onChange([
      ...selected,
      ...visible.filter((name) => !selected.includes(name)),
    ])
  const clearVisible = () =>
    onChange(selected.filter((name) => !visible.includes(name)))
  const isPatternTyped = allowPatterns && isToolPattern(entry)
  const isRefusedPattern = !allowPatterns && entry.includes("*")
  const add = () => {
    if (entry === "" || isRefusedPattern || selected.includes(entry)) return
    onChange([...selected, entry])
    setTyped("")
  }
  const inputLabel = allowPatterns
    ? "Add a tool name or pattern"
    : "Add a tool name"

  return (
    <div className="flex flex-col gap-3">
      {isKnown ? (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <SearchField
              label={`Search ${label.toLowerCase()}`}
              placeholder="Search tools"
              value={query}
              onChange={setQuery}
              isDisabled={isDisabled}
              className="min-w-0 flex-1"
            />
            <Button
              size="sm"
              onPress={selectVisible}
              isDisabled={isDisabled || visible.length === 0}
            >
              Select all
            </Button>
            <Button
              size="sm"
              onPress={clearVisible}
              isDisabled={
                isDisabled || !visible.some((name) => selected.includes(name))
              }
            >
              Select none
            </Button>
          </div>
          <p className="text-caption">
            {pickedKnown.length} of {known.length} {toolNoun(known.length)}{" "}
            selected
          </p>
          {known.length === 0 ? (
            <p className="text-caption">
              The server's last check listed no tools.
            </p>
          ) : visible.length === 0 ? (
            <p className="text-caption">No tool matches "{query.trim()}".</p>
          ) : (
            <ul
              aria-label={label}
              className="flex max-h-72 flex-col gap-1 overflow-y-auto"
            >
              {visible.map((name) => (
                <li key={name} className="min-w-0">
                  <Checkbox
                    isSelected={selected.includes(name)}
                    onChange={(next) => toggle(name, next)}
                    isDisabled={isDisabled}
                    className="min-h-11 md:min-h-0"
                  >
                    <span className="flex min-w-0 flex-col">
                      <span className="font-mono text-sm break-all">
                        {name}
                      </span>
                      {descriptionOf(name) ? (
                        <span className="text-caption break-words">
                          {descriptionOf(name)}
                        </span>
                      ) : null}
                    </span>
                  </Checkbox>
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-caption">
            Check the server to pick from its tools. Until then, type names
            {allowPatterns ? " or patterns" : ""} below.
          </p>
          {onCheck ? (
            <Button size="sm" onPress={onCheck} isPending={isChecking}>
              Check
            </Button>
          ) : null}
        </div>
      )}

      {others.length > 0 ? (
        <ul
          aria-label={`${label}: typed entries`}
          className="flex flex-col gap-1"
        >
          {others.map((name) => {
            const matches = toolsMatching(name, known).length
            const note = !isKnown
              ? ""
              : isToolPattern(name)
                ? matches === 0
                  ? "Matches no known tool"
                  : `Matches ${matches} known ${toolNoun(matches)}`
                : "Not offered by the server"
            const isFlagged = isKnown && matches === 0
            return (
              <li key={name} className="min-w-0">
                <Checkbox
                  isSelected
                  onChange={(next) => toggle(name, next)}
                  isDisabled={isDisabled}
                  className="min-h-11 md:min-h-0"
                >
                  <span className="flex flex-wrap items-center gap-x-2">
                    <span className="font-mono text-sm break-all">{name}</span>
                    {note ? (
                      <span
                        className={`flex items-center gap-1 text-caption ${isFlagged ? "text-warning" : ""}`}
                      >
                        {isFlagged ? <Dot className="bg-warning" /> : null}
                        {note}
                      </span>
                    ) : null}
                  </span>
                </Checkbox>
              </li>
            )
          })}
        </ul>
      ) : null}

      <div className="flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-1">
          <Field
            label={inputLabel}
            value={typed}
            onChange={setTyped}
            placeholder={allowPatterns ? "create_*" : "tool_name"}
            isDisabled={isDisabled}
            isInvalid={isRefusedPattern}
            errorMessage={
              isRefusedPattern ? "Only whole tool names go here." : undefined
            }
            description={
              isPatternTyped && isKnown
                ? `Matches ${toolsMatching(entry, known).length} of the server's ${known.length} ${toolNoun(known.length)}.`
                : undefined
            }
            shouldReserveMessage={false}
          />
        </div>
        <Button
          onPress={add}
          isDisabled={isDisabled || entry === "" || isRefusedPattern}
          className="min-h-11 md:min-h-0"
        >
          Add
        </Button>
      </div>
    </div>
  )
}
