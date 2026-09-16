import { Link } from "@tanstack/react-router"

// By its `@/…` specifier, never as `./overlayWalletSlot`: that specifier is the
// seam's alias key, and the module says what a relative import would cost.
import { WalletNavSlot } from "@/app/nav/overlayWalletSlot"
import { PLAYGROUND_NAV_ITEM } from "@/app/nav/registry"
import { useSurfaceVisibility } from "@/app/nav/useNavVisibility"
import { useDeployment } from "@/shared/hooks/useDeployment"

// Desktop destinations; the account menu keeps them reachable on mobile.
// WalletNavSlot is empty in OSS; the hosted overlay supplies its balance control.

const ACTION =
  "flex min-h-[2.125rem] items-center rounded-md px-1 text-shell-label font-medium text-muted transition-colors hover:text-foreground"

export function TopBarActions() {
  const { docs_url } = useDeployment()
  const hostsSurface = useSurfaceVisibility()

  return (
    // Hidden below the md breakpoint, where the mobile header has room for the
    // dismiss control and the trail and nothing else. The slot is inside the
    // cluster and so inherits that, which is what otari.ai's own navbar does
    // with the balance.
    <div className="hidden shrink-0 items-center gap-5 md:flex">
      {hostsSurface(PLAYGROUND_NAV_ITEM) && (
        <Link to={PLAYGROUND_NAV_ITEM.to} className={ACTION}>
          {PLAYGROUND_NAV_ITEM.label}
        </Link>
      )}
      {docs_url ? (
        <a
          href={docs_url}
          target="_blank"
          rel="noopener noreferrer"
          className={ACTION}
        >
          Documentation
        </a>
      ) : (
        <Link to="/docs" className={ACTION}>
          Documentation
        </Link>
      )}
      <WalletNavSlot />
    </div>
  )
}
