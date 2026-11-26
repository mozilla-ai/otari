import { Link } from "@heroui/react"
import type { ReactNode } from "react"
import { RouterProvider } from "react-aria-components"
import { navigateTo, routeHref } from "@/shared/helpers/appLocation"

import { LoginPageShell } from "./LoginPageShell"

/** Shared content and frame for every public authentication page. */
export function PublicAuthLayout({
  title,
  description,
  children,
  footer,
  isBusy = false,
}: {
  title: string
  /** Optional subhead under the title. */
  description?: ReactNode
  children: ReactNode
  /** Links below the divider: where to go next when this page is a dead end. */
  footer?: ReactNode
  /** Whether a request of the page's own is in flight, for the header slot. */
  isBusy?: boolean
}) {
  return (
    <LoginPageShell isBusy={isBusy}>
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
 * `to` is a route (`/recover-password`), not TanStack Router's `<Link to>`,
 * which the house style otherwise requires for an internal destination. These
 * pages are rendered by `DeploymentRoot` *ahead* of `RouterProvider` (see
 * `App.tsx`), so there is no TanStack context to link through. A react-aria
 * router scoped to this one link stands in: the anchor carries the real href,
 * so it opens in a new tab like any other, and a plain click moves the tab with
 * `navigateTo`, which `App` picks up and swaps the page in place rather than
 * reloading it. Scoped to the link rather than the page because the guide link
 * on `AuthHelp` is a HeroUI `Link` too, and it names a path the gateway serves
 * (`/welcome`) or another origin: the one link down here that *is* a page load.
 *
 * Sized to the 44px touch target the phone viewport asks for, which `text-sm`
 * alone is about half of, and these stack several deep in a card's footer.
 */
export function PublicAuthLink({
  to,
  children,
  isInline = false,
  onPress,
}: {
  to: string
  children: ReactNode
  /** Runs before the route changes, to hand something to the next page. */
  onPress?: () => void
  /**
   * For a link that shares a row with a label, where a 44px box would make the
   * row 44 tall. The row keeps the line's own 20px and the target is a bleed of
   * 16px above and 8px below, which stays clear of the field under it.
   */
  isInline?: boolean
}) {
  return (
    <RouterProvider navigate={followPublicAuthLink} useHref={routeHref}>
      <Link
        href={to}
        onPress={onPress}
        className={`inline-flex items-center text-sm font-medium text-link hover:text-link-hover ${
          isInline
            ? "relative before:absolute before:inset-x-0 before:-top-4 before:-bottom-2"
            : "min-h-11"
        }`}
      >
        {children}
      </Link>
    </RouterProvider>
  )
}

/** What a public auth link does when followed: move this tab to its route. */
function followPublicAuthLink(route: string): void {
  navigateTo(route)
}

/** Send this tab to another page in front of the session, from a handler. */
export function goToPublicAuthPage(to: string): void {
  navigateTo(to)
}

/**
 * Leave a page that must not be returned to, replacing its history entry so
 * Back skips it rather than replaying it.
 */
export function replacePublicAuthPage(to: string): void {
  navigateTo(to, { replace: true })
}
