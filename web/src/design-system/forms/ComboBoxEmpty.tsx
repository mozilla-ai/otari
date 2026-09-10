import type { ReactNode } from "react"

import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"

/**
 * What a combo box's popover says with nothing in it.
 *
 * Internal to the comboboxes the way `optionKey` is internal to the two
 * selects. Every one of them sets `allowsEmptyCollection`, so that a query
 * matching nothing keeps the popover open instead of closing the field under
 * the operator's hands, and an open popover therefore has to be able to say
 * nothing rather than show nothing. On the routing page that is the ordinary
 * case rather than an edge: model discovery reports nothing at all until a
 * provider credential exists.
 *
 * Two sentences, because they are two different facts. "Nothing matches what
 * you typed" is about the query, and typing less fixes it. "There is nothing to
 * offer yet" is about the source, and only the caller knows what would fill it,
 * which is why `emptyMessage` is where a feature says how.
 *
 * On `EmptyMessage`, the product's one "there is nothing here" treatment, held
 * to 44px so an empty popover is a box with a sentence in it rather than a
 * collapsed sliver.
 */
export function ComboBoxEmpty({
  isSourceEmpty,
  emptyMessage = "Nothing to choose from yet.",
  noMatchesMessage = "No matches. Try a shorter search.",
}: {
  /** True when the list is empty whatever is typed, so the source is what needs explaining. */
  isSourceEmpty?: boolean
  /** Shown while `isSourceEmpty`. Say what would fill the list. */
  emptyMessage?: ReactNode
  /** Shown when the source has options and the query matched none of them. */
  noMatchesMessage?: ReactNode
}) {
  return (
    <EmptyMessage minHeightClass="min-h-[2.75rem]">
      {isSourceEmpty ? emptyMessage : noMatchesMessage}
    </EmptyMessage>
  )
}
