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
 * No session is minted on accept: accepting only resolves their membership to
 * `active`, and the identity it resolves to is password-less on the roster
 * until it is claimed. So the page hands off to the claim flow (`#/signup`,
 * otari#835) with the invited address prefilled, rather than saying there is
 * nothing further to do, which was true only before per-user sign-in existed
 * and left every invitee stranded on the sign-in screen.
 *
 * Which ending the visitor gets is decided from what this browser and this
 * deployment can do, never from anything the server says about the address:
 * a signed-in visitor is already past the claim, and a gateway that cannot
 * send mail cannot run signup at all (`create_user_for_signup` refuses before
 * writing). Nothing here asks whether the invited identity has a password
 * yet, and nothing may: that is the enumeration answer `POST /v1/auth/signup`
 * withholds by design, so the claim and the sign-in are both offered and the
 * visitor picks.
 */

import { Button, Card, Link } from "@heroui/react"
import { useState } from "react"

import { useAuth } from "@/features/auth/AuthContext"
import { PublicAuthLink } from "@/features/auth/PublicAuthLayout"
import { useAcceptInvitation, useValidateInvitation } from "@/shared/api/hooks"
import { ErrorBanner } from "@/shared/components/ui"
import { tokenFromHash } from "@/shared/helpers/hashParams"
import { useDeployment } from "@/shared/hooks/useDeployment"

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
  const { isAuthenticated } = useAuth()
  // Signup is mail-gated for the reason `Login` gates its own link on the same
  // field (otari#648, and `publicAuthPaths`' table): the claim mails a
  // verification link and refuses with a 503 where this deployment has no
  // transport, so an invitation shared by hand on such a gateway has to say so
  // rather than offering a button whose only outcome is that refusal.
  const { mail_ready } = useDeployment()

  const goTo = (hash: string) => {
    window.location.hash = hash
  }

  // The address the invitation was sent to, still in the query cache behind the
  // accept: the claim is bound to it, so prefilling it is what keeps the
  // invitee from claiming a different address by mistyping this one. Absent
  // only if the preview were somehow gone, and then the form asks for it.
  const invitedEmail = preview.data?.email
  const signupHash = invitedEmail
    ? `#/signup?${new URLSearchParams({ email: invitedEmail }).toString()}`
    : "#/signup"

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
          ) : accept.isSuccess ? (
            // Ahead of the preview's own branches, because the preview refuses a
            // token that has been spent: a refetch after this accept (a
            // reconnect is enough) would otherwise replace what happened with
            // "already used" and take the next step away with it.
            <>
              {/* No article before the role: two of the three ("a owner", "a
                  admin") read wrong, and the roles are the server's words. */}
              <p className="text-center text-sm text-foreground">
                You're now a member of{" "}
                <strong>{accept.data.organization_name}</strong>, with the{" "}
                <strong>{accept.data.role}</strong> role.
              </p>
              {isAuthenticated ? (
                <>
                  <p className="text-center text-xs text-muted">
                    You're already signed in, so there is nothing left to set
                    up.
                  </p>
                  <Button
                    variant="primary"
                    fullWidth
                    onPress={() => goTo("#/")}
                  >
                    Go to the dashboard
                  </Button>
                </>
              ) : mail_ready ? (
                <>
                  <p className="text-center text-xs text-muted">
                    Next, set your password to sign in.
                  </p>
                  <Button
                    variant="primary"
                    fullWidth
                    onPress={() => goTo(signupHash)}
                  >
                    Set your password
                  </Button>
                  {/* The other half of the same fork, offered rather than
                      decided: an address that already has a password has
                      nothing to claim, and asking the server which case this is
                      would be the enumeration answer signup withholds. */}
                  <PublicAuthLink to="#/">
                    Already have a password? Sign in
                  </PublicAuthLink>
                </>
              ) : (
                <>
                  <p className="text-center text-xs text-muted">
                    Setting a password works by emailing you a link, and this
                    deployment is not configured to send mail. Ask whoever
                    administers it to set your password for you.
                  </p>
                  <Button
                    variant="primary"
                    fullWidth
                    onPress={() => goTo("#/")}
                  >
                    Go to sign in
                  </Button>
                </>
              )}
            </>
          ) : preview.isLoading ? (
            <p className="text-center text-sm text-muted">
              Checking your invitation…
            </p>
          ) : preview.error ? (
            <ErrorBanner error={preview.error} />
          ) : preview.data ? (
            <>
              <p className="text-center text-sm text-foreground">
                <strong>{preview.data.organization_name}</strong> has invited{" "}
                <strong>{preview.data.email}</strong> to join with the{" "}
                <strong>{preview.data.role}</strong> role.
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
