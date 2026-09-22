import { useState } from "react"

import type { OrgProviderKey } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { ComboBoxField } from "@/design-system/forms/ComboBoxField"
import {
  useOfferOrgProviderModel,
  useOrgProviderAvailableModels,
} from "@/shared/api/organizations"

// Offer one model on a key by name.
//
// The picker is what the provider says it serves on the stored credential,
// fetched only while this is open because answering means dialing the upstream.
// Custom values are allowed and the field degrades to a plain text box when the
// dial fails: neither the provider's list nor its silence is authoritative, and
// a backend that publishes no list is exactly the case this dialog exists for.
//
// No rate here. An organization's rates live in `organization_model_pricing`
// and are written through the rate editor on the page above, so a price set in
// two places could not disagree about what a request costs. A model offered
// with no rate anywhere is recorded and not served until one appears.

export function OfferModelDialog({
  providerKey,
  isOpen,
  onOpenChange,
  onOffered,
}: {
  providerKey: OrgProviderKey
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  onOffered: () => void
}) {
  const [model, setModel] = useState("")
  const [query, setQuery] = useState("")
  const available = useOrgProviderAvailableModels(providerKey.id, isOpen)
  const offer = useOfferOrgProviderModel(providerKey.id)

  // Filtered in the browser, unlike the Models page, and for a reason rather
  // than by omission: `/available-models` is one dial of the provider and
  // answers with its whole list, so there is no window here to be a prefix of.
  // Capped at 50 because a popover cannot render a thousand rows; the caller can
  // always type an id, which is sent exactly as typed.
  const listed = available.data?.models ?? []
  const matches = listed.filter(
    (id) => !query || id.toLowerCase().includes(query.toLowerCase()),
  )
  const options = matches.slice(0, 50).map((id) => ({ value: id, label: id }))

  // Three states behind one empty popover, each wanting a different sentence:
  // the dial is in flight, it came back refusing, or the provider genuinely
  // serves nothing. Only the middle one is worth an operator's attention.
  const emptyMessage = available.isLoading
    ? "Asking the provider what it serves…"
    : available.data?.error
      ? `${available.data.error} Type the model id; it is sent exactly as typed either way.`
      : available.data?.discovery_unsupported
        ? "This provider publishes no model list. Type the model id."
        : available.isError
          ? "The model list could not be read. Type the model id."
          : "The provider listed no models."

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      title={`Offer a model on ${providerKey.name}`}
      description={`Adds one model to what this organization serves through its ${providerKey.provider} key. It is priced from the community defaults, and stays unserved until something prices it.`}
      submitLabel="Offer model"
      isPending={offer.isPending}
      error={offer.error}
      isDirty={model.trim() !== ""}
      isSubmitDisabled={model.trim() === ""}
      onSubmit={() => {
        offer.mutate({ model: model.trim() }, { onSuccess: onOffered })
      }}
    >
      <ComboBoxField
        label="Model"
        value={model}
        onChange={setModel}
        onQueryChange={setQuery}
        options={options}
        allowsCustomValue
        isRequired
        isSourceEmpty={listed.length === 0}
        emptyMessage={emptyMessage}
        noMatchesMessage="No model the provider lists matches what you typed. It is offered exactly as typed."
        description="The id as the provider spells it, with no provider prefix."
      />
    </FormDialog>
  )
}
