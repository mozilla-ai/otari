import { Checkbox } from "@/design-system/forms/Checkbox"
import { useDeployment } from "@/shared/hooks/useDeployment"

/**
 * The box that accepts a deployment's published terms, with the links to read
 * them: the terms, and the privacy policy where the deployment publishes one.
 *
 * Rendered only where the bootstrap carries a `terms_url`, and required where
 * it is: an acceptance a page would have sent either way records nothing.
 *
 * Plain anchors and not router `Link`s, because the targets are addresses an
 * operator configured and usually off this origin, and beside the control
 * rather than inside its label, which is the only arrangement that lets them be
 * read: HTML exempts an interactive descendant from a label's own activation,
 * but react-aria presses the label from a document-level handler that knows no
 * such exemption and that nothing on the anchor can stop, so nested a link only
 * ticked the box (otari-ai#2146). `ariaLabel` carries the sentence the visible
 * label no longer holds in full.
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
  const { privacy_url } = useDeployment()
  return (
    // One run of text with a hanging indent of the box and its gap, so a
    // sentence that wraps on a phone continues under its own first word rather
    // than under the box. The box sits inline at the start of the first line.
    <p className="pl-6 -indent-6 text-caption">
      {/* 44px target from a pseudo-element bleed of 13 above and 12 below the
          first 20px line, so the box is easy to hit on a phone without
          `hasTouchTarget`, which would add 24px of height to the card. The
          bleed stays inside the 12px gaps every caller spaces it with. */}
      <span className="inline-block indent-0 align-top">
        <Checkbox
          isSelected={isAccepted}
          onChange={onChange}
          ariaLabel={
            privacy_url
              ? "I accept the Terms of Service and the Privacy Policy"
              : "I accept the Terms of Service"
          }
          className="relative before:absolute before:inset-x-0 before:-top-[13px] before:-bottom-[12px]"
        >
          <span className="text-caption">I accept the</span>
        </Checkbox>
      </span>{" "}
      <TermsLink href={termsUrl}>Terms of Service</TermsLink>
      {privacy_url ? (
        <>
          {" and the "}
          <TermsLink href={privacy_url}>Privacy Policy</TermsLink>
        </>
      ) : null}
      .
    </p>
  )
}

function TermsLink({ href, children }: { href: string; children: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="font-medium text-link hover:text-link-hover"
    >
      {children}
    </a>
  )
}
