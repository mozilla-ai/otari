import { PasswordCard } from "@/features/account/PasswordCard"
import { PageHeader } from "@/shared/components/ui"
import { useDeployment } from "@/shared/hooks/useDeployment"

/**
 * Account settings: what the signed-in identity can change about how it signs
 * in.
 *
 * One card today, and deliberately not folded into the Settings page, which is
 * the deployment's configuration: every row there is a gateway-wide setting an
 * operator changes on behalf of the process, while this is the credential that
 * identifies the person changing them. It is also the destination the account
 * menu has always named and could not open, and the surface the rest of #653's
 * auth affordances land on as their backends arrive: connected sign-in
 * providers (otari#651) and passkeys (otari#652).
 */
export function AccountPage() {
  const { session_type } = useDeployment()

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Account settings"
        description="How you sign in to this dashboard. Every setting here is yours alone; the gateway's own configuration is on the Settings page."
      />
      {session_type === "local_operator" ? (
        <PasswordCard />
      ) : (
        // A hosted session is minted by otari.ai and its credential is managed
        // there, so a password form here would write to the wrong control
        // plane. Said rather than left blank: an empty page reads as a page
        // that failed to load. The enum's third value, "none", cannot reach
        // this branch: only a hybrid gateway reports it, and `App` answers one
        // with the landing page instead of the router.
        <p className="text-sm text-muted">
          This deployment's sign-in is managed by the control plane that issued
          your session, so there is nothing to change here.
        </p>
      )}
    </div>
  )
}
