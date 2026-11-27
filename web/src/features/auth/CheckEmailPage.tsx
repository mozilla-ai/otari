import { useState } from "react"
import { routeSearchParams } from "@/shared/helpers/appLocation"
import { mailDelivers } from "@/shared/helpers/bootstrap"
import { useDeployment } from "@/shared/hooks/useDeployment"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"
import { readSignInAddress } from "./signInAddress"

/**
 * `/check-email`: where signup and a verification resend both land.
 *
 * It makes no request and knows nothing about the address beyond what the
 * visitor typed on the page before. That is the point: both routes behind it
 * answer the same sentence whether or not there was an identity to act on, so
 * this page has to be written in the same conditional or it would leak, on the
 * client, the fact the server declined to state. Naming the typed address
 * tells the visitor nothing the server said, only which inbox to open.
 *
 * Reading the deployment's mail state leaks nothing, because it is not a fact
 * about the address. Where email reaches no recipient, the page says to ask for
 * the link rather than telling the visitor to watch an inbox.
 *
 * `?type=resend` only changes which of the two sentences reads naturally; a
 * missing or unrecognized value falls back to the signup wording rather than
 * being validated, since the page is correct either way and a visitor who
 * hand-edited the URL should still get a page.
 */
export function CheckEmailPage({ route }: { route: string }) {
  const { mail } = useDeployment()
  const type = routeSearchParams(route).get("type")
  const [email] = useState(readSignInAddress)
  const address = email === "" ? "that address" : email
  const delivers = mailDelivers(mail)
  const link =
    type === "resend" ? "a new verification link" : "a verification link"
  // One clause for both states, so neither says more about the address than the
  // other.
  const condition =
    type === "resend"
      ? `If ${address} has an account waiting to be verified`
      : `If you can sign up with ${address}`

  return (
    <PublicAuthLayout
      title={delivers ? "Check your email" : "Ask for your link"}
      description={
        delivers
          ? `${condition}, we have sent it ${link}.`
          : `${condition}, ${link} was created for it. This deployment cannot email you a link that works, so ask whoever runs it for yours.`
      }
      footer={
        <>
          <PublicAuthLink to="/resend-verification">
            {delivers
              ? "Didn't get it? Send another link"
              : "Create another link"}
          </PublicAuthLink>
          <PublicAuthLink to="/">Back to sign in</PublicAuthLink>
        </>
      }
    >
      <p className="text-body">
        Open {delivers ? "the link in that message" : "the link"} to confirm the
        address. Signing in is blocked until you do.
      </p>
      <p className="text-caption">
        The link expires, and a new one can be {delivers ? "sent" : "created"}{" "}
        at any time.
      </p>
    </PublicAuthLayout>
  )
}
