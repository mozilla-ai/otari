import { Link } from "@heroui/react"
import { Popover } from "@/design-system/overlays/Popover"
import { PublicAuthLink } from "./PublicAuthLayout"

/**
 * The secondary actions a public auth page offers, behind one control.
 *
 * `credential` names the box the page in front of this one is asking for, and
 * is absent on a page that asks for no credential at all (`SignupPage`, which
 * is setting one rather than taking one). It decides two things: the note
 * saying what becomes of the credential, which has to name the one actually
 * typed, and where recovery is offered. `Login` keeps "Forgot your password?"
 * beside the password box itself, so this popover carries it only for the
 * master-key box, where there is no password field to put it next to.
 */
export function AuthHelp({
  offersRecovery,
  credential,
}: {
  offersRecovery: boolean
  credential?: "password" | "master-key"
}) {
  return (
    <Popover
      trigger={
        <span className="inline-flex min-h-11 items-center px-3 text-sm font-medium text-link">
          Help
        </span>
      }
      placement="top"
    >
      <div className="flex max-w-xs flex-col gap-2">
        {offersRecovery && credential === "master-key" ? (
          <PublicAuthLink to="#/recover-password">
            Forgot your password?
          </PublicAuthLink>
        ) : null}
        {offersRecovery ? (
          <PublicAuthLink to="#/resend-verification">
            Send a new verification link
          </PublicAuthLink>
        ) : null}
        <Link
          href="/welcome"
          className="inline-flex min-h-11 items-center text-sm font-medium text-link hover:text-link-hover"
        >
          Open the welcome guide
        </Link>
        {credential ? (
          <p className="border-t border-border pt-3 text-caption">
            {credential === "password" ? (
              "Your password is sent once and exchanged for a session cookie. It is never stored in the browser."
            ) : (
              <>
                Your{" "}
                <a href="/welcome" className="text-link hover:text-link-hover">
                  master key
                </a>{" "}
                is sent once and exchanged for a session cookie. It is never
                stored in the browser.
              </>
            )}
          </p>
        ) : null}
      </div>
    </Popover>
  )
}
