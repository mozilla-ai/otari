import { Link } from "@tanstack/react-router"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { isDeploymentOperator } from "@/features/organization/roles"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useProviders } from "@/shared/api/providers"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * Names the providers that serve requests on a credential environment variable
 * alone. Discovery lists configured providers only, so without this their
 * models are missing from the catalog with nothing saying why (#1626).
 *
 * Operator-only: the list describes the gateway's own environment, and
 * `/providers` refuses anyone else.
 */
export function EnvOnlyProvidersNotice() {
  const organization = useOrganizationContext()
  const isOperator = isDeploymentOperator(organization.data)
  const providers = useProviders(isOperator)
  const hostsProvidersPage = useSurfaces()("providers")

  // Read through `isOperator`: a disabled query still hands back what sits
  // under its key, so a caller demoted mid-session would keep the notice.
  const names = isOperator ? (providers.data?.env_only_providers ?? []) : []
  if (names.length === 0) {
    return null
  }

  const one = names.length === 1
  return (
    <InfoBanner>
      <code>{names.join(", ")}</code> {one ? "serves" : "serve"} requests
      through {one ? "its" : "their"} credential environment variable, but{" "}
      {one ? "its" : "their"} models are not listed here: only configured
      providers are. Add {one ? "it" : "them"} under <code>providers:</code> in
      config.yml
      {hostsProvidersPage ? (
        <>
          {" "}
          or on the{" "}
          <Link to="/providers" className="underline underline-offset-2">
            Providers page
          </Link>
        </>
      ) : null}{" "}
      to list {one ? "its" : "their"} models.
    </InfoBanner>
  )
}
