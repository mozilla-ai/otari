import { Link } from "@tanstack/react-router"

import { EntitlementGate } from "@/shared/components/EntitlementGate"
import { useDeployment } from "@/shared/hooks/useDeployment"

// The right end of the top bar: the links that are not destinations in either
// rail.
//
// Documentation is the guide bundled with this gateway. It sits in the chrome
// because it is read alongside a page rather than instead of one, and the
// design's account menu has no row for it; the menu keeps a row of its own
// anyway, because this cluster is hidden below `md` and the guide would
// otherwise have no entry point on a phone.
//
// Playground is a hosted surface, gated on the two things it actually needs:
// it is a page otari.ai serves and this gateway does not, so the link needs
// both the entitlement and a `management_url` to point at, which only a gateway
// attached to otari.ai has.
//
// The design also draws a balance here. It is not built: this gateway meters
// spend but holds no wallet, and there is no seam that could feed one, since an
// overlay replaces `overlaySections.ts` / `overlayLabelOverrides.ts` and not
// this file. A prop and a component that nothing can reach read as wired, so
// they wait for the deployment that has a figure to report.

const ACTION =
  "flex min-h-[2.125rem] items-center rounded-md px-1 text-chrome-row font-medium text-muted transition-colors hover:text-foreground"

export function TopBarActions() {
  const { management_url } = useDeployment()
  const platform = management_url?.replace(/\/$/, "")

  return (
    // Hidden below the md breakpoint, where the mobile header has room for the
    // dismiss control and the trail and nothing else.
    <div className="hidden shrink-0 items-center gap-5 md:flex">
      <Link to="/docs" className={ACTION}>
        Documentation
      </Link>
      {platform ? (
        <EntitlementGate capability="playground">
          <a
            href={`${platform}/playground`}
            target="_blank"
            rel="noopener noreferrer"
            className={ACTION}
          >
            Playground
          </a>
        </EntitlementGate>
      ) : null}
    </div>
  )
}
