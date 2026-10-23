import { Fragment } from "react"

import { navContextForPath, pageLabelForPath } from "@/app/nav/registry"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/**
 * Where you are, at the head of the content pane: scope first, then the page.
 *
 * The organization does not lead the trail. One is provisioned at first boot and
 * most deployments keep exactly that, so naming it on every page is a segment
 * that never disambiguates anything; where a caller does belong to a second,
 * the scope switcher above the rail names the active one.
 *
 * The three contexts have different scopes to show: the workspace rail is inside
 * one workspace, and the organization rail is not inside any.
 */
export function Breadcrumbs({ pathname }: { pathname: string }) {
  const organization = useOrganizationContext()
  const { selected } = useSelectedWorkspace()

  const page = pageLabelForPath(pathname)
  const organizationName = organization.data?.organization?.name
  // A destination no rail owns names itself and no scope. The trail would
  // otherwise read "Default workspace / Settings" and file the deployment's own
  // settings under a workspace that does not own them, which is the same claim
  // the rail used to make before `ORG_PATHS` stopped counting these.
  //
  // The scope goes rather than the page. Dropping the page here would empty the
  // trail on a deployment with no workspace selected (a first run has none),
  // leaving a reader with neither a crumb nor a highlighted rail row, and no
  // location at all is worse than a location that is merely brief.
  const context = navContextForPath(pathname)

  const trail = [
    // The deployment has no name to show. The other two scopes are data (a
    // workspace's name, an organization's), and the process is the one every
    // tenant shares, so the word is the scope: "Deployment / Settings".
    context === "deployment"
      ? "Deployment"
      : context === "organization"
        ? organizationName
        : selected?.name,
    page,
  ].filter((one): one is string => Boolean(one))

  // Deduped so a page labeled like its scope is not named twice.
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
