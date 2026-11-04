import { useState } from "react"
import { FiChevronRight } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { SearchField } from "@/design-system/forms/SearchField"
import { DismissChip } from "@/design-system/indicators/DismissChip"
import { groupModelsByVendor } from "@/features/workspaces/access/modelVendors"

type Mode = "all" | "only"

const MODE_OPTIONS = [
  { value: "all", label: "All models, including ones added later" },
  { value: "only", label: "Only these models" },
] as const

/**
 * Which of one provider's models a workspace may call, in a dialog.
 *
 * The models sit under their vendors, because a provider that routes to others
 * serves several (Together serves DeepSeek, Meta and Qwen models), and a
 * vendor's box selects all of its models or shows that some are. A vendor's
 * models open beneath it; a search opens every vendor it matches.
 *
 * `models` is null where nothing lists what the provider serves, and the names
 * are typed instead. `initial` and what `onSave` receives are null for every
 * model, including ones added later. Waits for Save: a list is built over
 * several presses, and an empty one is unfinished rather than a block, which
 * is what the row's switch is for.
 */
export function ModelSelectionDialog({
  isOpen,
  onClose,
  title,
  models,
  vendors,
  initial,
  onSave,
  isPending,
  error,
  validateEntry = () => undefined,
}: {
  isOpen: boolean
  onClose: () => void
  title: string
  models: readonly string[] | null
  vendors: ReadonlyMap<string, string>
  initial: readonly string[] | null
  onSave: (next: string[] | null) => void
  isPending: boolean
  error: unknown
  /** Why a typed name cannot be added, or undefined when it can. */
  validateEntry?: (name: string) => string | undefined
}) {
  const [mode, setMode] = useState<Mode>(initial === null ? "all" : "only")
  const [picked, setPicked] = useState<ReadonlySet<string>>(
    new Set(initial ?? []),
  )
  const [query, setQuery] = useState("")
  const [opened, setOpened] = useState<ReadonlySet<string>>(new Set())
  const [draft, setDraft] = useState("")

  const isIncomplete = mode === "only" && picked.size === 0
  const initialSet = new Set(initial ?? [])
  const isDirty =
    (mode === "all") !== (initial === null) ||
    (mode === "only" &&
      (picked.size !== initialSet.size ||
        [...picked].some((model) => !initialSet.has(model))))

  const setMany = (names: readonly string[], isSelected: boolean) =>
    setPicked((current) => {
      const next = new Set(current)
      for (const name of names) {
        if (isSelected) next.add(name)
        else next.delete(name)
      }
      return next
    })

  // A model picked before the provider stopped listing it stays visible, so it
  // can still be taken off.
  const listed =
    models === null ? [] : [...new Set([...models, ...picked])].sort()
  const needle = query.trim().toLowerCase()
  const matching = needle
    ? listed.filter(
        (model) =>
          model.toLowerCase().includes(needle) ||
          (vendors.get(model) ?? "").toLowerCase().includes(needle),
      )
    : listed
  const groups = groupModelsByVendor(matching, vendors)

  const trimmed = draft.trim()
  const entryProblem =
    trimmed === ""
      ? undefined
      : picked.has(trimmed)
        ? "This model is already in the list."
        : validateEntry(trimmed)
  const addDraft = () => {
    if (trimmed === "" || entryProblem !== undefined) return
    setMany([trimmed], true)
    setDraft("")
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={title}
      description="Which of this provider's models requests billed to this workspace may call."
      size="lg"
      submitLabel="Save"
      onSubmit={() => onSave(mode === "all" ? null : [...picked].sort())}
      isPending={isPending}
      error={error}
      isDirty={isDirty}
      isSubmitDisabled={isIncomplete}
      footerStart={
        mode === "only" ? (
          <span className="text-caption">
            {picked.size} {picked.size === 1 ? "model" : "models"} selected
          </span>
        ) : null
      }
    >
      <RadioGroup
        label="Models"
        hideLabel
        value={mode}
        onChange={(next) => setMode(next === "only" ? "only" : "all")}
        options={MODE_OPTIONS}
      />
      {mode === "only" && models === null ? (
        <div className="flex flex-col gap-2">
          <span className="text-caption">
            Nothing lists this provider's models, so type each name as the
            provider spells it.
          </span>
          {picked.size > 0 ? (
            <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 pointer-coarse:gap-y-5">
              {[...picked].sort().map((model) => (
                <li key={model}>
                  <DismissChip
                    value={model}
                    dismissLabel={`Remove ${model}`}
                    onDismiss={() => setMany([model], false)}
                  />
                </li>
              ))}
            </ul>
          ) : null}
          {/* Enter here means Add, not Save: the dialog is a real form, and
              its implicit submission would save without the typed name. */}
          <fieldset
            aria-label="Add a model"
            className="flex min-w-0 items-start gap-2"
            onKeyDown={(event) => {
              if (event.key !== "Enter" || event.metaKey || event.ctrlKey)
                return
              event.preventDefault()
              addDraft()
            }}
          >
            <Field
              label="Model name"
              value={draft}
              onChange={setDraft}
              placeholder="As the provider spells it"
              isInvalid={entryProblem !== undefined}
              errorMessage={entryProblem}
              shouldReserveMessage={false}
            />
            <Button
              className="mt-7"
              isDisabled={trimmed === "" || entryProblem !== undefined}
              onPress={addDraft}
            >
              Add
            </Button>
          </fieldset>
        </div>
      ) : null}
      {mode === "only" && models !== null ? (
        listed.length === 0 ? (
          <span className="text-caption">
            The deployment serves no models on this provider right now.
          </span>
        ) : (
          <div className="flex flex-col gap-3">
            <SearchField
              label="Search models"
              placeholder="Search models or vendors"
              value={query}
              onChange={setQuery}
            />
            {groups.length === 0 ? (
              <span className="text-caption">No model matches.</span>
            ) : (
              <ul className="flex flex-col divide-y divide-border-subtle border border-border">
                {groups.map((group) => {
                  const selected = group.models.filter((model) =>
                    picked.has(model),
                  ).length
                  // A search, or a provider with one vendor, leaves nothing to
                  // hide behind the chevron.
                  const isOpen =
                    needle !== "" ||
                    groups.length === 1 ||
                    opened.has(group.vendor)
                  return (
                    <li key={group.vendor} className="flex flex-col">
                      <div className="flex min-h-11 items-center gap-1 pr-3 pl-1">
                        <Button
                          isIconOnly
                          size="sm"
                          aria-label={`${isOpen ? "Hide" : "Show"} ${group.vendor} models`}
                          aria-expanded={isOpen}
                          onPress={() =>
                            setOpened((current) => {
                              const next = new Set(current)
                              if (!next.delete(group.vendor))
                                next.add(group.vendor)
                              return next
                            })
                          }
                        >
                          <FiChevronRight
                            aria-hidden
                            className={`size-4 text-subtle transition-transform motion-reduce:transition-none ${
                              isOpen ? "rotate-90" : ""
                            }`}
                          />
                        </Button>
                        <Checkbox
                          className="min-w-0 flex-1"
                          ariaLabel={`All ${group.vendor} models`}
                          isSelected={selected === group.models.length}
                          isIndeterminate={
                            selected > 0 && selected < group.models.length
                          }
                          onChange={(isSelected) =>
                            setMany(group.models, isSelected)
                          }
                        >
                          <span className="text-emphasis">{group.vendor}</span>
                        </Checkbox>
                        <span className="text-caption text-subtle">
                          {selected} of {group.models.length}
                        </span>
                      </div>
                      {isOpen ? (
                        <ul className="flex flex-col border-t border-border-subtle bg-surface-subtle py-1 pl-12">
                          {group.models.map((model) => (
                            <li key={model}>
                              <Checkbox
                                hasTouchTarget
                                className="justify-start"
                                isSelected={picked.has(model)}
                                onChange={(isSelected) =>
                                  setMany([model], isSelected)
                                }
                              >
                                <span className="text-mono-caption">
                                  {model}
                                </span>
                              </Checkbox>
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        )
      ) : null}
      {isIncomplete ? (
        <span className="text-caption">
          Pick at least one model. To stop this workspace using the provider,
          switch it off instead.
        </span>
      ) : null}
    </FormDialog>
  )
}
