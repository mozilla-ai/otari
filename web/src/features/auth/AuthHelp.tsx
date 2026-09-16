import { Link } from "@heroui/react"
import { Popover } from "@/design-system/overlays/Popover"
import { PublicAuthLink } from "./PublicAuthLayout"

export function AuthHelp({
  offersRecovery,
  usesPassword,
}: {
  offersRecovery: boolean
  usesPassword?: boolean
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
        {offersRecovery && usesPassword === false ? (
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
        {usesPassword !== undefined ? (
          <p className="border-t border-border pt-3 text-caption">
            {usesPassword ? (
              "Your password is sent once and exchanged for a session cookie. It is never stored in the browser."
            ) : (
              <>
                Your{" "}
                <a href="/welcome" className="text-link">
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
