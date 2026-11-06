import { Link } from "@heroui/react"
import { Button } from "@/design-system/actions/Button"
import { Popover } from "@/design-system/overlays/Popover"
import { useDeployment } from "@/shared/hooks/useDeployment"
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
 *
 * The guide is the gateway's own `/welcome` page, which a deployment that
 * configures `docs_url` (otari.ai) does not serve at its origin, so there the
 * link goes to that documentation instead of a dead path.
 *
 * The trigger is a `Button` because a popover trigger has to be a react-aria
 * pressable, and reads as the text link beside it because both call sites put it
 * in an `.otari-auth-actions` row (`globals.css`). Outside one of those rows it
 * gets the default ghost box back.
 */
export function AuthHelp({
  offersRecovery,
  credential,
}: {
  offersRecovery: boolean
  credential?: "password" | "master-key"
}) {
  const { docs_url } = useDeployment()
  const link = "text-link hover:text-link-hover"
  return (
    <Popover
      label="Help"
      trigger={<Button variant="ghost">Help</Button>}
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
          href={docs_url ?? "/welcome"}
          {...(docs_url
            ? { target: "_blank", rel: "noopener noreferrer" }
            : {})}
          className={`inline-flex min-h-11 items-center text-sm font-medium ${link}`}
        >
          {docs_url ? "Read the documentation" : "Open the welcome guide"}
        </Link>
        {credential ? (
          <p className="border-t border-border pt-3 text-caption">
            {credential === "password" ? (
              "Your password is sent once and exchanged for a session cookie. It is never stored in the browser."
            ) : (
              <>
                Your{" "}
                <a
                  href={docs_url ?? "/welcome"}
                  {...(docs_url
                    ? { target: "_blank", rel: "noopener noreferrer" }
                    : {})}
                  className={link}
                >
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
