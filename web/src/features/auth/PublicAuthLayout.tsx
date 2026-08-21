import { Card, Link } from "@heroui/react"
import type { ReactNode } from "react"

/**
 * The card every page in front of a session renders into.
 *
 * The platform's `PreloginLayout` is the ancestor, but three of the four
 * things it does are hosted-product chrome that has no counterpart here: the
 * Mozilla.ai byline, the marketing footer, and the pin that forces the auth
 * flow to the light theme regardless of the visitor's preference. A
 * self-hosted dashboard is an operator tool whose theme is the operator's
 * choice, so what survives the port is the shape `Login` and
 * `AcceptInvitationPage` already established here: one centered card, the
 * mark, a heading, and a body.
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
    <div className="flex min-h-full items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <Card.Content className="flex flex-col gap-5 p-7">
          <div className="flex flex-col items-center gap-3 text-center">
            <img src="/favicon.svg" alt="Otari" className="h-12 w-12" />
            <div>
              <h1 className="text-lg font-semibold text-foreground">{title}</h1>
              {description ? (
                <p className="mt-1 text-sm text-muted">{description}</p>
              ) : null}
            </div>
          </div>

          {children}

          {footer ? (
            <div className="flex flex-col items-center gap-2 border-t border-border pt-4 text-center">
              {footer}
            </div>
          ) : null}
        </Card.Content>
      </Card>
    </div>
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
 * picks it up and swaps the page in place. `Login` and `AcceptInvitationPage`
 * already reach for this component for their own out-of-router link to
 * `/welcome`.
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
      className="text-sm font-medium text-link hover:text-link-hover"
    >
      {children}
    </Link>
  )
}

/** Send this tab to another page in front of the session, from a handler. */
export function goToPublicAuthPage(to: string): void {
  window.location.hash = to
}
