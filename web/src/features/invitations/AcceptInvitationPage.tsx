/**
 * The page an emailed accept link lands on: `#/accept-invitation?token=...`.
 *
 * Deliberately not one of `src/routes/`'s files. Every route there lives
 * behind `App.tsx`'s auth gate (`DeploymentRoot` renders `<Login/>` instead
 * of the router when a session is required and absent), and the whole point
 * of this page is that the recipient holds neither a session nor the master
 * key. `App.tsx` renders this component directly, ahead of that gate, the
 * same way it renders `<Login/>` as a plain component rather than a route.
 *
 * No session is minted on accept: Otari has no per-user sign-in yet, so
 * there is nothing to sign this visitor into. Accepting only resolves their
 * membership to `active`; they reach the sign-in screen next, the same as
 * anyone else added to an organization before that flow exists.
 */

import { Button, Card, Link } from "@heroui/react"
import { useState } from "react"

import { useAcceptInvitation, useValidateInvitation } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import { tokenFromHash } from "@/shared/helpers/hashParams"

export function AcceptInvitationPage() {
  // Read once, safely: App.tsx renders this keyed on the hash
  // (`<AcceptInvitationPage key={hash} />`), so a *different* invitation link
  // opened in the same tab remounts a fresh instance of this component rather
  // than re-rendering this one with a stale token frozen in its initial
  // state. Without that key, a same-type re-render on hashchange would keep
  // this state (and its token) exactly as it was.
  const [token] = useState(() => tokenFromHash(window.location.hash))
  const preview = useValidateInvitation(token ?? "")
  const accept = useAcceptInvitation()

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <Card.Content className="flex flex-col gap-5 p-7">
          <div className="flex flex-col items-center gap-3 text-center">
            <img src="/favicon.svg" alt="Otari" className="h-12 w-12" />
            <h1 className="text-lg font-semibold text-foreground">
              Organization invitation
            </h1>
          </div>

          {token === null ? (
            <ErrorBanner
              error={
                new Error(
                  "This link is missing its invitation token, so there is nothing to accept.",
                )
              }
            />
          ) : preview.isLoading ? (
            <p className="text-center text-sm text-muted">
              Checking your invitation…
            </p>
          ) : preview.error ? (
            <ErrorBanner error={preview.error} />
          ) : accept.isSuccess ? (
            <>
              <p className="text-center text-sm text-foreground">
                You're now a member of{" "}
                <strong>{accept.data.organization_name}</strong> as a{" "}
                <strong>{accept.data.role}</strong>.
              </p>
              <p className="text-center text-xs text-muted">
                There is no sign-in for this identity yet, so there is nothing
                further to do here.
              </p>
              <Button
                variant="primary"
                fullWidth
                onPress={() => {
                  window.location.hash = "#/"
                }}
              >
                Continue
              </Button>
            </>
          ) : preview.data ? (
            <>
              <p className="text-center text-sm text-foreground">
                <strong>{preview.data.organization_name}</strong> has invited{" "}
                <strong>{preview.data.email}</strong> to join as a{" "}
                <strong>{preview.data.role}</strong>.
              </p>
              <ErrorBanner error={accept.error} />
              <Button
                variant="primary"
                fullWidth
                isPending={accept.isPending}
                onPress={() => {
                  if (token) accept.mutate(token)
                }}
              >
                Accept invitation
              </Button>
            </>
          ) : null}

          <div className="border-t border-border pt-4 text-center">
            <Link
              href="/welcome"
              className="text-sm font-medium text-link hover:text-link-hover"
            >
              New to Otari? Open the welcome guide
            </Link>
          </div>
        </Card.Content>
      </Card>
    </div>
  )
}
