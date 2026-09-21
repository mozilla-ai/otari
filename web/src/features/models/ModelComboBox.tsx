import { type ReactNode, useMemo } from "react"
import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"
import { isPrefixedSelector } from "@/features/models/modelKey"
import { useCatalogSearch, useDiscoverableModels } from "@/shared/api/models"
import { useDebounced } from "@/shared/hooks/useDebounced"

// How many matches to render at once. A single provider can report a few hundred
// models, and past this the popover is a wall of text nobody scrolls; typing one
// more character is faster. How many were withheld is reported rather than the
// list silently ending.
const MAX_VISIBLE = 50

/** Where a picker's suggestions come from. */
export type ModelSource = "discovery" | "catalog"

/**
 * The model source behind the picker: the options to offer and what to say about
 * them.
 *
 * Exported because the hint is a sentence, and a sentence cannot live inside a
 * control row: a field whose message wraps is taller than its siblings, and a
 * row laid out `items-end` then floats that field's input line above theirs.
 * `web/design/forms.md` names that break and its remedy, which is to put the
 * message under the row rather than inside one of its fields. A caller doing
 * that reads the hint from here and renders it itself.
 *
 * `description` is deliberately not a parameter: the last branch returns
 * `undefined` and the caller falls back to its own, so this answers only what
 * the source has to say.
 */
export function useModelCatalog(
  value: string,
  source: ModelSource = "discovery",
) {
  const isDiscovery = source === "discovery"
  const discoverable = useDiscoverableModels(isDiscovery)
  // The catalog's matches, chosen by the server. Debounced on the way into the
  // key, or every keystroke is a request and the answers race (otari#1380).
  // Discovery still filters what it was given: `/v1/models/discoverable` takes
  // no search term, and it lists what a provider reports rather than what this
  // deployment serves, so the catalog is not a stand-in for it. #1380 carries
  // the reasoning.
  const term = useDebounced(value.trim())
  const catalog = useCatalogSearch(term, MAX_VISIBLE + 1, !isDiscovery)

  const { visible, total, failed, isSourceEmpty } = useMemo(() => {
    const query = value.trim().toLowerCase()
    const providers = isDiscovery ? (discoverable.data?.providers ?? []) : []
    if (!isDiscovery) {
      // One row per selector, because a selector is what the field submits and
      // the catalog folds several of them onto one model. The narrowing is the
      // server's; this only unfolds what it sent, keeping the selectors that
      // answer the term so a matched model does not drag its siblings in.
      const selectors = (catalog.data?.models ?? []).flatMap((model) =>
        model.selectors.filter(
          (selector) =>
            isPrefixedSelector(selector) &&
            (!query || selector.toLowerCase().includes(query)),
        ),
      )
      const options: ComboBoxOption[] = [...new Set(selectors)]
        .sort((a, b) => a.localeCompare(b))
        .map((selector) => ({ value: selector, label: selector }))
      return {
        visible: options.slice(0, MAX_VISIBLE),
        // The server's count of matching models, which is what the hint is
        // about: the rows withheld are models, not selectors of one model.
        total: Math.max(options.length, catalog.data?.count ?? 0),
        failed: [],
        isSourceEmpty: !term && options.length === 0,
      }
    }
    // Provider order is preserved, so rows still cluster by provider even
    // without section headers.
    const all: ComboBoxOption[] = providers.flatMap((provider) =>
      provider.models.map((model) => ({
        value: model.key,
        label: model.key,
      })),
    )
    const hits = query
      ? all.filter((option) => option.value.toLowerCase().includes(query))
      : all
    return {
      visible: hits.slice(0, MAX_VISIBLE),
      total: hits.length,
      failed: providers.filter((provider) => !provider.ok),
      isSourceEmpty: all.length === 0,
    }
  }, [catalog.data, discoverable.data, isDiscovery, term, value])

  // `isLoading` rather than the `isPending && !data` the rest of the dashboard
  // guards on, and for the property that rules it out there: it is false for a
  // disabled query. The other source is always disabled, so `isPending` would
  // have this saying "Loading" about a read it is deliberately not making.
  const isLoading = isDiscovery ? discoverable.isLoading : catalog.isLoading
  // A refused or failed read leaves the same empty list as a gateway with no
  // provider, and the two want opposite things said: one is filled by adding a
  // credential, the other by nothing the operator can do from this form.
  const isRefused = isDiscovery ? discoverable.isError : catalog.isError

  // Four states behind one empty popover, and each wants a different sentence:
  // the list is still coming, it could not be read, nothing has been configured
  // to fill it, or nothing this caller can route to fills it.
  const emptyMessage = isLoading
    ? "Looking for models…"
    : isRefused
      ? "The model list could not be read. Type the selector; it is sent exactly as typed either way."
      : isDiscovery
        ? "No models discovered yet. Add a provider credential and the models it serves appear here; until then, type the selector."
        : "No models to offer yet. Models this deployment serves appear here once a provider is configured; until then, type the selector."

  const hint = ((): ReactNode => {
    if (isLoading) {
      return isDiscovery
        ? "Loading models from your providers…"
        : "Loading the model catalog…"
    }
    if (isRefused) {
      return "Could not read the model list. Type the selector directly."
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
    return undefined
  })()

  return { visible, isSourceEmpty, emptyMessage, hint }
}

/**
 * Model selector backed by the deployment's model discovery or by the catalog.
 *
 * Free text is always allowed: neither source is authoritative. Discovery only
 * sees what the configured credentials expose and the catalog only what this
 * caller could route to, so a model behind an unconfigured provider, a
 * brand-new release, or a backend that cannot list must stay typeable. The
 * dropdown is a shortcut, never a whitelist.
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
  isInvalid,
  errorMessage,
  source = "discovery",
  hintPlacement = "field",
  describedBy,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  description?: ReactNode
  placeholder?: string
  autoFocus?: boolean
  isRequired?: boolean
  isInvalid?: boolean
  /**
   * The correction, announced on the input and rendered under it beside the
   * hint rather than in place of it: the hint line is already spoken for while
   * a search is being narrowed ("Showing 50 of 210 matches"), and a form that
   * put the correction there would lose it exactly when the operator is typing.
   */
  errorMessage?: string
  /**
   * Where the suggestions come from.
   *
   * `"discovery"` reads GET /v1/models/discoverable, which is a
   * deployment-operator read: an organization admin who does not operate the
   * deployment is refused it (#821). `"catalog"` reads GET /v1/models, which
   * answers any signed-in caller and is already narrowed server-side to what
   * that caller could route to, so it is what a tenant-facing form asks for.
   * The two are composed with `enabled` rather than branched on at the call
   * site, so only the chosen one is ever requested.
   */
  source?: ModelSource
  /**
   * Where the catalog hint goes. `"field"` renders it under the input, which is
   * right for a field standing on its own. `"detached"` renders nothing and
   * leaves the caption line empty, for a caller inside a control row that
   * renders the hint under the whole row instead; it reads the text from
   * `useModelCatalog`.
   *
   * Named for the placement rather than the hint, because `useModelCatalog`
   * returns a `hint` too and that one is the sentence.
   */
  hintPlacement?: "field" | "detached"
  /** Id of the detached hint, so it is still announced with this input. */
  describedBy?: string
}) {
  const { visible, isSourceEmpty, emptyMessage, hint } = useModelCatalog(
    value,
    source,
  )

  return (
    <ComboBoxField
      label={label}
      value={value}
      onChange={onChange}
      options={visible}
      description={
        hintPlacement === "detached" ? undefined : (hint ?? description)
      }
      // Only while the hint exists: `SectionRow` renders the span carrying this
      // id conditionally, so passing it unconditionally pointed five inputs at
      // nothing in the ordinary state, where the catalog lists cleanly.
      describedBy={hint ? describedBy : undefined}
      placeholder={placeholder}
      autoFocus={autoFocus}
      isRequired={isRequired}
      isInvalid={isInvalid}
      errorMessage={errorMessage}
      // Discovery is not authoritative, so anything typed stands on its own.
      allowsCustomValue
      // Not "focus", which with autoFocus drops the whole list open as soon as
      // the form appears. React Aria marks everything outside an open popover
      // aria-hidden, so the price fields and Save button would be unreachable
      // to a screen reader before a single keystroke.
      menuTrigger="input"
      isSourceEmpty={isSourceEmpty}
      emptyMessage={emptyMessage}
      noMatchesMessage="No model matches. Type a provider:model selector to use it anyway."
    />
  )
}
