import {
  ComboBox,
  Description,
  Input,
  Label,
  ListBox,
  ListBoxItem,
} from "@heroui/react"
import { useState } from "react"
import { FiSearch } from "react-icons/fi"
import { ComboBoxEmpty } from "@/design-system/forms/ComboBoxEmpty"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import {
  limitStatus,
  limitWords,
  matchTargets,
  type RuleTarget,
  SEPARATOR,
  targetRowText,
} from "@/features/rules/ruleTargets"
import type { ReadState } from "@/features/spend/useSpend"

// The rows are already the search's matches, on more than the name, so
// react-aria must not filter them again by the row's text.
const KEEP_EVERY_ROW = () => true

/**
 * One of a kind, searched by name: the second half of "Applies to". The rows
 * say who each one is and the limit they already have; the line under the
 * field says the same of the one chosen, and keeps its height while empty.
 */
export function RuleTargetPicker({
  label,
  placeholder,
  targets,
  value,
  onChange,
  emptyWords,
  limits,
  isDisabled,
}: {
  label: string
  placeholder: string
  /** Every target of the kind; the field matches them against what is typed. */
  targets: readonly RuleTarget[]
  /** The chosen target's `value`, or "" for none. */
  value: string
  onChange: (value: string) => void
  /** What the open list says with no rows, given what is typed. */
  emptyWords: (query: string) => string
  /** The read behind each target's limit. */
  limits?: ReadState
  isDisabled?: boolean
}) {
  const selected = targets.find((one) => one.value === value)
  // What is being typed, until a row is picked or the field is left.
  const [typed, setTyped] = useState<string>()
  const shown = matchTargets(targets, typed ?? "")

  return (
    <ComboBox.Root
      allowsEmptyCollection
      menuTrigger="focus"
      defaultFilter={KEEP_EVERY_ROW}
      // Both halves controlled, so a pick is reported once and the input shows
      // the chosen name rather than what was typed to find it.
      inputValue={typed ?? selected?.name ?? ""}
      selectedKey={selected && typed === undefined ? selected.value : null}
      onInputChange={(next) => {
        setTyped(next)
        // An emptied box is how a choice is cleared.
        if (next === "") onChange("")
      }}
      onSelectionChange={(key) => {
        setTyped(undefined)
        if (key != null) onChange(String(key))
      }}
      isDisabled={isDisabled}
      className="otari-rule-target flex min-w-0 flex-col gap-1"
    >
      <Label className="text-body">{label}</Label>
      <ComboBox.InputGroup className="relative">
        <FiSearch
          aria-hidden
          className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted"
        />
        <Input
          placeholder={placeholder}
          autoComplete="off"
          data-1p-ignore
          data-lpignore="true"
          className="truncate"
        />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover className="otari-rule-target__list">
        <ListBox
          items={shown}
          className="max-h-72 overflow-auto"
          renderEmptyState={() => (
            <ComboBoxEmpty noMatchesMessage={emptyWords(typed ?? "")} />
          )}
        >
          {(target: RuleTarget) => (
            <ListBoxItem
              id={target.value}
              // The name alone: it is what the input shows once picked.
              textValue={target.name}
              aria-label={targetRowText(target)}
            >
              <span className="flex min-w-0 flex-1 items-center justify-between gap-3">
                <span className="flex min-w-0 flex-col">
                  <span className="truncate">{target.name}</span>
                  {target.detail ? (
                    <span className="truncate text-caption text-subtle">
                      {target.detail}
                    </span>
                  ) : null}
                </span>
                {target.own ? (
                  <span className="shrink-0 text-caption tabular-nums">
                    {limitWords(target.own)}
                  </span>
                ) : null}
              </span>
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
      <FieldMessages>
        {/* Always mounted: react-aria wires a description to the input when
            the field mounts, and one that arrived with the choice would be
            shown and never announced. One line at any width: who it is gives
            way, the limit does not. */}
        <Description className="flex min-w-0 text-muted tabular-nums">
          {selected?.detail ? (
            <>
              <span className="truncate">{selected.detail}</span>
              <span className="shrink-0 whitespace-pre">{SEPARATOR}</span>
            </>
          ) : null}
          {selected ? (
            <span className="shrink-0">{limitStatus(selected, limits)}</span>
          ) : null}
        </Description>
      </FieldMessages>
    </ComboBox.Root>
  )
}
