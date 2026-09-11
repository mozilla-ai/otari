import { Link } from "@heroui/react"
import type { ComponentType, ReactNode } from "react"

/** The left-aligned frame for recovery, verification, and other public pages. */
export function AuthPageShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-full flex-col">
      <header className="flex h-14 shrink-0 items-center border-b border-border px-4 md:px-6">
        {/* The real mark, so the tab icon and the page agree. `alt=""` because
            nothing here is a destination and the heading below names the
            product. */}
        <img
          src={`${import.meta.env.BASE_URL}favicon.svg`}
          alt=""
          className="h-6 w-[26px]"
        />
      </header>
      <div className="flex min-h-0 flex-1">
        {/* `min-h-full` on the column is what runs the rule the height of the
            page even when its content is short, which is what makes the ground
            beyond read as ground rather than as the page having ended. */}
        <div className="flex w-full max-w-[520px] shrink-0 flex-col gap-6 border-r border-border px-4 py-10 md:px-6">
          {children}
        </div>
      </div>
    </div>
  )
}

/** Shared auth content, with a replaceable frame for login and signup. */
export function PublicAuthLayout({
  title,
  description,
  children,
  footer,
  shell: Shell = AuthPageShell,
}: {
  title: string
  shell?: ComponentType<{ children: ReactNode }>
  /** Optional subhead under the title. */
  description?: ReactNode
  children: ReactNode
  /** Links below the divider: where to go next when this page is a dead end. */
  footer?: ReactNode
}) {
  return (
    <Shell>
      <div className="flex flex-col gap-1.5">
        <h1 className="text-display">{title}</h1>
        {description ? (
          <p className="text-sm text-pretty text-muted">{description}</p>
        ) : null}
      </div>

      {children}

      {footer ? (
        <div className="flex flex-col border-t border-border pt-5">
          {footer}
        </div>
      ) : null}
    </Shell>
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
 * picks it up and swaps the page in place. The `/welcome` links on `Login` and
 * `AcceptInvitationPage` are a HeroUI `Link` of their own rather than this,
 * and stay that way: `/welcome` is a real path the gateway serves, so it is
 * the one link down here that *is* a page load.
 *
 * Sized to the 44px touch target the phone viewport asks for, which `text-sm`
 * alone is about half of, and these stack several deep in a card's footer.
 */
export function PublicAuthLink({
  to,
  children,
}: {
  to: string
  children: ReactNode
}) {
  return (
    <Link
      href={to}
      className="inline-flex min-h-11 items-center text-sm font-medium text-link hover:text-link-hover"
    >
      {children}
    </Link>
  )
}

/** Send this tab to another page in front of the session, from a handler. */
export function goToPublicAuthPage(to: string): void {
  window.location.hash = to
}
