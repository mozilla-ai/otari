import { Checkbox } from "@/design-system/forms/Checkbox"

/**
 * The box that accepts a deployment's published terms, with the link to read them.
 *
 * Rendered only where the bootstrap carries a `terms_url`, and required where
 * it is: an acceptance a page would have sent either way records nothing.
 *
 * A plain anchor and not a router `Link`, because the target is an address an
 * operator configured and is usually off this origin, and beside the control
 * rather than inside its label, which is the only arrangement that lets the
 * terms be read: HTML exempts an interactive descendant from a label's own
 * activation, but react-aria presses the label from a document-level handler
 * that knows no such exemption and that nothing on the anchor can stop, so
 * nested the link only ticked the box (otari-ai#2146). `ariaLabel` carries the
 * sentence the visible label no longer holds in full.
 */
export function TermsAcceptanceField({
  termsUrl,
  isAccepted,
  onChange,
}: {
  termsUrl: string
  isAccepted: boolean
  onChange: (isAccepted: boolean) => void
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-1 text-caption">
      {/* 44px target from a pseudo-element bleed of 13 above and 12 below the
          19px row, so the box is easy to hit on a phone without
          `hasTouchTarget`, which would add 25px of height to the card. The
          bleed stays inside the 12px gaps every caller spaces it with. */}
      <Checkbox
        isSelected={isAccepted}
        onChange={onChange}
        ariaLabel="I accept the terms of service"
        className="relative before:absolute before:inset-x-0 before:-top-[13px] before:-bottom-[12px]"
      >
        <span className="text-caption">I accept the</span>
      </Checkbox>
      <span>
        <a
          href={termsUrl}
          target="_blank"
          rel="noreferrer"
          className="font-medium text-link hover:text-link-hover"
        >
          terms of service
        </a>
        .
      </span>
    </div>
  )
}
