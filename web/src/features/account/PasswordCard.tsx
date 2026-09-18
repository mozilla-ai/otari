import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { Section } from "@/design-system/layout/Section"
import { useOrganizationContext } from "@/shared/api/organizations"

import { PasswordDialog } from "./PasswordDialog"
import {
  passwordAction,
  passwordFormShape,
  passwordSummary,
} from "./passwordForm"

/**
 * The password this identity signs in to the dashboard with: set it for the
 * first time, or change it.
 *
 * One endpoint (`PUT /v1/auth/password`) behind three readings of one form, and
 * which one applies is read off the *caller*, not off the deployment. That is
 * the correction: `sign_in_methods` describes the gateway, and the question
 * here is what the person signed in right now holds, which `caller.has_password`,
 * `caller.claims_deployment` and `caller.email` on the membership context answer.
 *
 * - **Claiming.** First boot leaves the operator identity with no address and
 *   no password, and the master key as the dashboard login. Setting the
 *   operator's password is the single act that retires master-key sign-in on
 *   this deployment (mozilla-ai/otari-ai#1716). An operator adopted from an
 *   existing tenancy already has an address, so only a password is asked for.
 * - **Setting a first password.** Somebody who signs in through Google, GitHub
 *   or a passkey holds no password, and neither does a roster entry nobody has
 *   claimed. They have an address already, so only a new password is asked for.
 *   Keying this off the deployment is what used to strand them: a claimed
 *   deployment showed everybody the change form, which asks for a current
 *   password they could never supply and offered no other way through
 *   (mozilla-ai/otari-ai#2099).
 * - **Changing.** From then on the server requires the current password from a
 *   cookie-authenticated caller.
 *
 * The form is a dialog rather than three fields sitting open on the page. It is
 * a credential change reached by deliberate act, it is the one thing on this
 * page that is not safe to half-fill and wander away from, and the card can then
 * say in a line what the account currently signs in with.
 *
 * A successful call moves two things this component does not own. The address
 * and the new `has_password` are seated back onto the membership context by
 * `useSetPassword`, so this card re-reads its own shape rather than keeping a
 * mode of its own; and a claim reports itself through `useRetireMasterKeySignIn`,
 * because the bootstrap is a context read once per load and no invalidation
 * reaches it.
 */
export function PasswordCard() {
  const context = useOrganizationContext()
  const caller = context.data?.caller
  const [isOpen, setIsOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so a draft credential is
  // cleared on the way in rather than left in memory on the way out
  // (`SpendCeilingsCard` is the pattern).
  const [openCount, setOpenCount] = useState(0)
  // What the last successful call did. Neither fact survives it otherwise: the
  // address comes back in the response, and whether that call was the claim
  // cannot be read off the shape afterwards, since claiming is what changes it.
  const [outcome, setOutcome] = useState<{
    email: string
    isClaimed: boolean
  }>()

  const shape = caller ? passwordFormShape(caller) : undefined

  return (
    <Section
      aria-labelledby="account-password-title"
      className="border-t border-border pt-6 pb-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 id="account-password-title" className="text-title">
        Dashboard password
      </h2>

      {context.isPending && !context.data ? (
        <Skeleton
          className="h-20 w-full max-w-md"
          ariaLabel="Loading your sign-in details"
        />
      ) : !shape ? (
        // No guessed form. Which of the three readings applies is a fact about
        // the signed-in identity, and a form built on the wrong guess is what
        // this card was fixing: the change form asks for a password an OAuth
        // sign-in never had, and the claim form's address is refused for
        // anybody who already holds one.
        <>
          <p className="max-w-3xl text-sm text-muted">
            Who is signed in could not be read, so there is nothing here to
            change yet.
          </p>
          <ErrorBanner error={context.error} />
        </>
      ) : (
        <>
          <p className="max-w-3xl text-sm text-muted">
            {passwordSummary(shape)}
          </p>

          {outcome ? (
            <p
              role="status"
              aria-live="polite"
              className="max-w-3xl text-sm text-success"
            >
              {outcome.isClaimed
                ? `Saved. Sign in as ${outcome.email} from now on: the master key no longer signs in to this dashboard, and it stays the credential for the management API.`
                : `Saved. Your other sessions have ended; sign in as ${outcome.email} next time.`}
            </p>
          ) : null}

          <div>
            <Button
              variant="primary"
              onPress={() => {
                // The saved line describes a call that is not the one about to
                // be made, so it goes with the form that replaces it.
                setOutcome(undefined)
                setOpenCount((count) => count + 1)
                setIsOpen(true)
              }}
            >
              {passwordAction(shape)}
            </Button>
          </div>

          <PasswordDialog
            key={openCount}
            isOpen={isOpen}
            onOpenChange={setIsOpen}
            shape={shape}
            onSaved={setOutcome}
          />
        </>
      )}
    </Section>
  )
}
