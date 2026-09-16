import type { CallerIdentity } from "@/client"

/**
 * What the signed-in identity asks for, which is the whole of what the password
 * form has to render.
 *
 * Both flags are the caller's own state and neither is the deployment's, which
 * is the correction behind mozilla-ai/otari-ai#2099: `sign_in_methods` says
 * which credentials this gateway accepts, and the form turns on which one the
 * person reading it holds. They are also exactly the two conditions
 * `PUT /v1/auth/password` branches on, so a form built from them cannot ask for
 * a field the gateway ignores or omit one it requires.
 *
 * Shared by the card and the dialog rather than derived twice: the control that
 * opens the form and the one that completes it have to say the same words, and
 * `FormDialog` makes that a contract.
 */
export interface PasswordFormShape {
  /** The identity has no sign-in address, so this call has to supply one. */
  needsEmail: boolean
  /** The identity holds a password, so replacing it means proving the old one. */
  needsCurrentPassword: boolean
}

export function passwordFormShape(caller: CallerIdentity): PasswordFormShape {
  return {
    needsEmail: !caller.email,
    needsCurrentPassword: caller.has_password,
  }
}

/** The card's line about what this account signs in with today. */
export function passwordSummary({
  needsEmail,
  needsCurrentPassword,
}: PasswordFormShape): string {
  if (needsCurrentPassword) {
    return "The password you sign in to this dashboard with. Changing it ends every other session this identity holds; this one stays signed in."
  }
  if (needsEmail) {
    return "This gateway still signs in with its master key. Set an address and a password to sign in as yourself from now on. The master key stays the credential for the management API, and it can still reset this password if you forget it."
  }
  return "You have no dashboard password. You signed in another way, through a connected account or a passkey, and that keeps working: a password is a second way in, and it is what lets you sign in where those are not available."
}

/** The dialog's one line under its title: the consequence, not the title again. */
export function passwordDialogDescription({
  needsEmail,
  needsCurrentPassword,
}: PasswordFormShape): string {
  if (needsCurrentPassword) {
    return "You stay signed in here; every other session this identity holds ends."
  }
  if (needsEmail) {
    return "The address and password this deployment signs in with from now on. The master key stops being a dashboard login."
  }
  return "A second way in, beside however you sign in now. Every other session this identity holds ends."
}

/** The label the card's trigger and the dialog's submit both carry. */
export function passwordAction({
  needsEmail,
  needsCurrentPassword,
}: PasswordFormShape): string {
  if (needsCurrentPassword) {
    return "Change password"
  }
  return needsEmail ? "Claim this deployment" : "Set a password"
}
