import { Fragment } from "react"

import { navContextForPath, navLabelForPath } from "@/app/nav/registry"
import { useOrganizationContext } from "@/shared/api/hooks"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment } from "@/shared/hooks/useDeployment"

/**
 * Where you are, at the head of the content pane: scope first, then the page.
 *
 * The organization is left out of a standalone deployment's trail. One is
 * provisioned at first boot and most deployments keep exactly that, so naming it
 * on every page is a segment that never disambiguates anything; where a caller
 * does belong to a second, the scope switcher above the rail names the active
 * one. A hosted deployment can hold several, so there it leads.
 *
 * The two contexts have different scopes to show: the workspace rail is inside
 * one workspace, and the organization rail is not inside any.
 */
export function Breadcrumbs({ pathname }: { pathname: string }) {
  const { deployment_type } = useDeployment()
  const organization = useOrganizationContext()
  const { selected } = useSelectedWorkspace()

  const page = navLabelForPath(pathname)
  const organizationName = organization.data?.organization?.name
  const inOrganization = navContextForPath(pathname) === "organization"

  const trail = [
    deployment_type === "standalone" ? undefined : organizationName,
    inOrganization ? organizationName : selected?.name,
    page,
  ].filter((one): one is string => Boolean(one))

  // Deduped because the two rules above can name the organization twice on a
  // hosted deployment's organization rail.
  const segments = trail.filter((one, index) => trail.indexOf(one) === index)
  if (segments.length === 0) return null

  return (
    <nav aria-label="Breadcrumb" className="flex min-w-0 items-center">
      <ol className="flex min-w-0 items-center gap-1.5 text-sm">
        {segments.map((segment, index) => (
          <Fragment key={segment}>
            {index > 0 ? (
              <li aria-hidden="true" className="text-muted">
                /
              </li>
            ) : null}
            <li
              className={
                index === segments.length - 1
                  ? "truncate font-medium text-foreground"
                  : "truncate text-muted"
              }
              aria-current={index === segments.length - 1 ? "page" : undefined}
            >
              {segment}
            </li>
          </Fragment>
        ))}
      </ol>
    </nav>
  )
}
