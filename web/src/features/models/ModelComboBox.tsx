import { type ReactNode, useMemo } from "react"
import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"
import { useDiscoverableModels } from "@/shared/api/models"

// How many matches to render at once. A single provider can report a few hundred
// models, and past this the popover is a wall of text nobody scrolls; typing one
// more character is faster. How many were withheld is reported rather than the
// list silently ending.
const MAX_VISIBLE = 50

/**
 * Model selector backed by GET /v1/models/discoverable.
 *
 * Free text is always allowed: discovery only sees what the configured
 * credentials expose, so a model behind an unconfigured provider, a brand-new
 * release, or a backend that cannot list must stay typeable. The dropdown is a
 * shortcut, never a whitelist.
 *
 * Everything a provider reports is offered, unfiltered. any-llm's model type
 * carries no capability field, so there is no honest way to tell a chat model
 * from an embedding or TTS one here, and guessing from name patterns would hide
 * real models. Search is the answer instead.
 *
 * Rows are flat, each showing the full `provider:model` selector, rather than
 * grouped under provider headers. React Aria's sectioned collections
 * (ListBoxSection/Collection) throw "childNodes is not supported" under React
 * 19.2's development build, whose performance-track logging walks props and
 * trips over React Aria's synthetic collection DOM. The uncaught error breaks
 * the commit, so picking an option silently fails to update the field.
 * Production builds are unaffected, but a picker that only works in prod is not
 * worth the headers, and the provider is legible on every row regardless.
 *
 * Nothing discovered is the ordinary state of a gateway with no provider
 * credential, not an edge case, so the empty popover says which of the two it
 * is and what fills it. The wording names the credential rather than the page
 * that holds one: a hosted deployment keeps those under the organization
 * instead of the process-wide providers page.
 */
export function ModelComboBox({
  label,
  value,
  onChange,
  description,
  placeholder = "provider:model",
  autoFocus,
  isRequired,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  description?: ReactNode
  placeholder?: string
  autoFocus?: boolean
  isRequired?: boolean
}) {
  const discoverable = useDiscoverableModels()

  const { visible, total, failed, isCatalogEmpty } = useMemo(() => {
    const query = value.trim().toLowerCase()
    const providers = discoverable.data?.providers ?? []
    // Provider order is preserved, so rows still cluster by provider even
    // without section headers.
    const all: ComboBoxOption[] = providers.flatMap((provider) =>
      provider.models.map((model) => ({ value: model.key, label: model.key })),
    )
    const hits = query
      ? all.filter((option) => option.value.toLowerCase().includes(query))
      : all
    return {
      visible: hits.slice(0, MAX_VISIBLE),
      total: hits.length,
      failed: providers.filter((provider) => !provider.ok),
      isCatalogEmpty: all.length === 0,
    }
  }, [discoverable.data, value])

  const hint = ((): ReactNode => {
    if (discoverable.isLoading) {
      return "Loading models from your providers…"
    }
    // A failed provider is worth saying out loud: its models are simply absent
    // from the list, which is indistinguishable from a provider that has none.
    if (failed.length > 0) {
      const names = failed.map((provider) => provider.provider).join(", ")
      return `Could not list models for ${names}. Check that provider's credentials, or type the model key directly.`
    }
    if (total > visible.length) {
      return `Showing ${visible.length} of ${total} matches. Keep typing to narrow them.`
    }
    return description
  })()

  return (
    <ComboBoxField
      label={label}
      value={value}
      onChange={onChange}
      options={visible}
      description={hint}
      placeholder={placeholder}
      autoFocus={autoFocus}
      isRequired={isRequired}
      // Discovery is not authoritative, so anything typed stands on its own.
      allowsCustomValue
      // Not "focus", which with autoFocus drops the whole list open as soon as
      // the form appears. React Aria marks everything outside an open popover
      // aria-hidden, so the price fields and Save button would be unreachable
      // to a screen reader before a single keystroke.
      menuTrigger="input"
      isSourceEmpty={isCatalogEmpty}
      emptyMessage={
        discoverable.isLoading
          ? "Looking for models…"
          : "No models discovered yet. Add a provider credential and the models it serves appear here; until then, type the selector."
      }
      noMatchesMessage="No model matches. Type a provider:model selector to use it anyway."
    />
  )
}
