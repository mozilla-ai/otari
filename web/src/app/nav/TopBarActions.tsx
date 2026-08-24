import { Link } from "@tanstack/react-router"

// By its `@/…` specifier, never as `./overlayWalletSlot`: that specifier is the
// seam's alias key, and the module says what a relative import would cost.
import { WalletNavSlot } from "@/app/nav/overlayWalletSlot"
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
// The design also draws a balance at the end of the cluster, and this build has
// none to draw: this gateway meters spend but holds no wallet. `WalletNavSlot`
// is the seam a build that does hold one replaces to contribute the chip, so
// the gap is reachable rather than something an overlay would have to edit this
// file to fill. It renders nothing here.

const ACTION =
  "flex min-h-[2.125rem] items-center rounded-md px-1 text-chrome-row font-medium text-muted transition-colors hover:text-foreground"

export function TopBarActions() {
  const { management_url } = useDeployment()
  const platform = management_url?.replace(/\/$/, "")

  return (
    // Hidden below the md breakpoint, where the mobile header has room for the
    // dismiss control and the trail and nothing else. The slot is inside the
    // cluster and so inherits that, which is what otari.ai's own navbar does
    // with the balance.
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
      <WalletNavSlot />
    </div>
  )
}
