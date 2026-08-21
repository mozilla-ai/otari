import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"

/**
 * `#/check-email`: where signup and a verification resend both land.
 *
 * It makes no request and knows nothing about the address. That is the point:
 * both routes behind it answer the same sentence whether or not there was an
 * identity to act on, so this page has to be written in the same conditional
 * or it would leak, on the client, the fact the server declined to state.
 *
 * `?type=resend` only changes which of the two sentences reads naturally; a
 * missing or unrecognized value falls back to the signup wording rather than
 * being validated, since the page is correct either way and a visitor who
 * hand-edited the URL should still get a page.
 */
export function CheckEmailPage({ hash }: { hash: string }) {
  const type = new URLSearchParams(hash.split("?")[1] ?? "").get("type")

  return (
    <PublicAuthLayout
      title="Check your email"
      description={
        type === "resend"
          ? "If that address is registered and still unverified, a fresh verification link is on its way."
          : "If that address is on this gateway's roster and unclaimed, a verification link is on its way."
      }
      footer={
        <>
          <PublicAuthLink to="#/resend-verification">
            Didn't get it? Send another link
          </PublicAuthLink>
          <PublicAuthLink to="#/">Back to sign in</PublicAuthLink>
        </>
      }
    >
      <p className="text-center text-sm text-foreground">
        Open the link in that message to confirm the address. Signing in is
        blocked until you do.
      </p>
      <p className="text-center text-xs text-muted">
        The link expires, and a new one can be sent at any time.
      </p>
    </PublicAuthLayout>
  )
}
