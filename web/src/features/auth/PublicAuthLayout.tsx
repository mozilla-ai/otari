import { Link } from "@heroui/react"
import type { ReactNode } from "react"

/**
 * The sparse page: a 56px bar carrying the mark, a column pinned to the left
 * with a full-height rule down its right, and empty ground beyond it.
 *
 * Pinned rather than centered, and that is the whole of the pattern. The
 * signed-in pages put their content in a column starting at the page's left
 * gutter, so a centered card meant the page moved sideways at the moment of
 * signing in; this way the column the eye is reading is in the same place
 * before and after. The ground to the right is deliberately empty: there is one
 * thing to do on this screen.
 *
 * Exported because `Login` renders the same shell and the two must not drift.
 * `Login` has three states with different bodies and so composes the shell
 * itself; every other page in front of a session goes through
 * `PublicAuthLayout` below.
 */
export function AuthPageShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-full flex-col">
      <header className="flex h-14 shrink-0 items-center border-b border-border px-4 md:px-6">
        {/* The real mark, so the tab icon and the page agree. `alt=""` because
            nothing here is a destination and the heading below names the
            product. */}
        <img src="/favicon.svg" alt="" className="h-6 w-[26px]" />
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

/**
 * Every page in front of a session, in the shell above.
 *
 * The platform's `PreloginLayout` is the ancestor, but three of the four
 * things it does are hosted-product chrome that has no counterpart here: the
 * Mozilla.ai byline, the marketing footer, and the pin that forces the auth
 * flow to the light theme regardless of the visitor's preference. A
 * self-hosted dashboard is an operator tool whose theme is the operator's
 * choice, so what survives the port is a heading, a body, and where to go next.
 */
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
    <AuthPageShell>
      {/* The mark is on the bar, so the title stands on its own and the column
          starts at its left edge like every other column in the product. */}
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
    </AuthPageShell>
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
