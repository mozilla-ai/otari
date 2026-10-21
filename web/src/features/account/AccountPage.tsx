import { PageIntro } from "@/design-system/layout/PageIntro"
import { PasskeysCard } from "@/features/account/PasskeysCard"
import { PasswordCard } from "@/features/account/PasswordCard"
import { ProfileCard } from "@/features/account/ProfileCard"
import { useDeployment } from "@/shared/hooks/useDeployment"

/**
 * Account settings: what the signed-in identity can change about itself, which
 * is its name and how it signs in.
 *
 * Deliberately not folded into the Settings page, which is the deployment's
 * configuration: every row there is a gateway-wide setting an operator changes
 * on behalf of the process, while these are the credentials that identify the
 * person changing them. It is also the destination the account menu has always
 * named and could not open, and the surface the rest of #653's auth affordances
 * land on as their backends arrive: passkeys (otari#652) have landed, and
 * connected sign-in providers (otari#651) are still to come.
 */
export function AccountPage() {
  const { passkeys_enabled } = useDeployment()

  return (
    <div className="flex flex-col">
      <PageIntro title="Account settings">
        Who you are on this deployment, and how you sign in to it. Every setting
        here is yours alone; the gateway&rsquo;s own configuration is on the
        Settings page.
      </PageIntro>
      <ProfileCard />
      <PasswordCard />
      {/* Absent where the deployment turned passkeys off: the card would ask
          routes that are not there. A deployment that merely has not set them
          up still shows it, saying so. */}
      {passkeys_enabled ? <PasskeysCard /> : null}
    </div>
  )
}
