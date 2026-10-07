import { Link } from "@heroui/react"
import type { ReactNode } from "react"

import { LoginPageShell } from "./LoginPageShell"

/** Shared content and frame for every public authentication page. */
export function PublicAuthLayout({
  title,
  description,
  children,
  footer,
}: {
  title: string
  /** Optional subhead under the title. */
  description?: ReactNode
  children: ReactNode
  /** Links below the divider: where to go next when this page is a dead end. */
  footer?: ReactNode
}) {
  return (
    <LoginPageShell>
      <div className="flex flex-col gap-1.5 text-center">
        <h1 className="text-display">{title}</h1>
        {description ? (
          <p className="text-sm text-pretty text-muted">{description}</p>
        ) : null}
      </div>

      {children}

      {footer ? (
        <div className="flex flex-col border-t border-border pt-2">
          {footer}
        </div>
      ) : null}
    </LoginPageShell>
  )
}

/**
 * A link between two pages that both live in front of the router.
 *
 * The `href` is a hash path, not TanStack Router's `<Link to>`, which the
 * house style otherwise requires for an internal destination. These pages are
 * rendered by `DeploymentRoot` *ahead* of `RouterProvider` (see `App.tsx`), so
 * there is no router context to link through; and a hash change is not the
 * full page reload that rule exists to prevent, because `App`'s `useHashPath`
 * picks it up and swaps the page in place. The `/welcome` links on `AuthHelp`
 * and `AcceptInvitationPage` are a HeroUI `Link` of their own rather than this,
 * and stay that way: on the deployments that offer it at all, `/welcome` is a
 * path the gateway serves, so it is the one link down here that *is* a page
 * load. Which deployments those are is `welcomeGuideHref`'s answer.
 *
 * Sized to the 44px touch target the phone viewport asks for, which `text-sm`
 * alone is about half of, and these stack several deep in a card's footer.
 */
export function PublicAuthLink({
  to,
  children,
  isInline = false,
}: {
  to: string
  children: ReactNode
  /**
   * For a link that shares a row with a label, where a 44px box would make the
   * row 44 tall. The row keeps the line's own 20px and the target is a bleed of
   * 16px above and 8px below, which stays clear of the field under it.
   */
  isInline?: boolean
}) {
  return (
    <Link
      href={to}
      className={`inline-flex items-center text-sm font-medium text-link hover:text-link-hover ${
        isInline
          ? "relative before:absolute before:inset-x-0 before:-top-4 before:-bottom-2"
          : "min-h-11"
      }`}
    >
      {children}
    </Link>
  )
}

/** Send this tab to another page in front of the session, from a handler. */
export function goToPublicAuthPage(to: string): void {
  window.location.hash = to
}
