import { ComboBox, Input, ListBox, ListBoxItem } from "@heroui/react"
import { type ReactNode, useMemo, useState } from "react"
import { ComboBoxEmpty } from "@/design-system/forms/ComboBoxEmpty"
import { ControlField } from "@/design-system/forms/FieldMessages"
import { DismissChip } from "@/design-system/indicators/DismissChip"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"

// An access-list is a tri-state:
//   undefined -> "inherit" (`null` on the wire: unrestricted, or the owner's)
//   []        -> "block"   (deny all)
//   list      -> "only"    (restrict to these entries)
// A bare multi-select cannot tell "inherit" from "block" (both look empty), so
// the mode is an explicit 3-way choice. "Only" with no entries is an
// INCOMPLETE form (Save disabled), never a silent deny-all.
type Mode = "inherit" | "only" | "block"

function modeOf(entries: string[] | undefined): Mode {
  if (!entries) return "inherit"
  return entries.length === 0 ? "block" : "only"
}

export interface AccessOption {
  // The value stored on the list.
  id: string
  // What the operator sees in the dropdown.
  label: string
}

export interface AccessListCopy {
  inherit: string
  only: string
  block: string
  // The notice under "block", after "Blocked from ".
  blocked: ReactNode
  pickPrompt: string
  inputLabel: string
  placeholder: string
}

const MAX_VISIBLE = 50

// The shell both the model and the search tool access-lists share; each
// wrapper builds its own catalog and copy, and says what is still loading.
export function AccessListControl({
  initial,
  onChange,
  title,
  description,
  copy,
  catalog,
  hint,
  loadingMessage,
  showPicker = true,
}: {
  initial: string[] | undefined
  onChange: (value: string[] | undefined, valid: boolean) => void
  title: ReactNode
  description: ReactNode
  copy: AccessListCopy
  catalog: AccessOption[]
  hint: string | null
  // Said in the empty dropdown while the catalog is still being read.
  loadingMessage: string | null
  showPicker?: boolean
}) {
  const [mode, setMode] = useState<Mode>(modeOf(initial))
  const [entries, setEntries] = useState<string[]>(initial ?? [])
  const [query, setQuery] = useState("")

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return catalog
      .filter((option) => !entries.includes(option.id))
      .filter(
        (option) =>
          !q ||
          option.id.toLowerCase().includes(q) ||
          option.label.toLowerCase().includes(q),
      )
      .slice(0, MAX_VISIBLE)
  }, [catalog, entries, query])

  const emit = (nextMode: Mode, nextEntries: string[]) => {
    if (nextMode === "inherit") onChange(undefined, true)
    else if (nextMode === "block") onChange([], true)
    else onChange(nextEntries, nextEntries.length > 0)
  }

  const chooseMode = (next: Mode) => {
    setMode(next)
    emit(next, entries)
  }

  const addEntry = (id: string) => {
    const next = entries.includes(id) ? entries : [...entries, id]
    setEntries(next)
    setQuery("")
    emit("only", next)
  }

  const removeEntry = (id: string) => {
    const next = entries.filter((entry) => entry !== id)
    setEntries(next)
    emit("only", next)
  }

  // The shared tab, which is what every segmented choice in the product uses. A
  // raised chip (`bg-surface` plus an elevation that is `none` now shadows are
  // zeroed) leaves the selected segment reading 1.1:1 against its own track.
  const modeButton = (value: Mode, label: string) => (
    <Tab
      key={value}
      isActive={mode === value}
      onPress={() => chooseMode(value)}
    >
      {label}
    </Tab>
  )

  return (
    <div className="flex flex-col gap-3">
      <ControlField label={title} description={description} />
      {/* No track: a tab row is bare, and the selected segment's own fill is
          what marks it. A track plus a fill was two ways of saying one thing. */}
      <TabRow>
        {modeButton("inherit", copy.inherit)}
        {modeButton("only", copy.only)}
        {modeButton("block", copy.block)}
      </TabRow>

      {mode === "block" ? (
        <div className="flex items-center gap-2 border-y border-border py-2 text-xs text-muted">
          Blocked from {copy.blocked}
        </div>
      ) : null}

      {mode === "only" ? (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap gap-1.5">
            {entries.length === 0 ? (
              <span className="text-caption">{copy.pickPrompt}</span>
            ) : (
              entries.map((entry) => (
                <DismissChip
                  key={entry}
                  value={entry}
                  onDismiss={() => removeEntry(entry)}
                  dismissLabel={`Remove ${entry}`}
                />
              ))
            )}
          </div>
          {hint ? <span className="text-caption">{hint}</span> : null}
          {showPicker ? (
            <ComboBox.Root
              allowsEmptyCollection
              menuTrigger="input"
              inputValue={query}
              onInputChange={setQuery}
              selectedKey={null}
              onSelectionChange={(key) => {
                if (key != null) addEntry(String(key))
              }}
              // Cap the width so the field and its dropdown trigger stay within
              // easy reach instead of stretching across a wide form.
              className="flex max-w-md flex-col gap-1"
            >
              <ComboBox.InputGroup>
                <Input
                  aria-label={copy.inputLabel}
                  placeholder={copy.placeholder}
                  autoComplete="off"
                />
                <ComboBox.Trigger />
              </ComboBox.InputGroup>
              <ComboBox.Popover>
                <ListBox
                  items={visible}
                  className="max-h-72 overflow-auto"
                  renderEmptyState={() => (
                    <ComboBoxEmpty
                      isSourceEmpty={catalog.every((option) =>
                        entries.includes(option.id),
                      )}
                      emptyMessage={
                        loadingMessage ??
                        "Everything available is already on the list."
                      }
                      noMatchesMessage="Nothing matches what you typed."
                    />
                  )}
                >
                  {(option: AccessOption) => (
                    <ListBoxItem id={option.id} textValue={option.label}>
                      {option.label}
                    </ListBoxItem>
                  )}
                </ListBox>
              </ComboBox.Popover>
            </ComboBox.Root>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
