import { buttonVariants } from "@heroui/react"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { ModelCatalogView } from "@/features/models/ModelCatalogPage"
import { ModelDetailView } from "@/features/models/ModelDetailPage"
import { publicCatalogHref } from "@/features/models/publicCatalog"
import { useDeployment } from "@/shared/hooks/useDeployment"

// The catalog ahead of a session, where the deployment has opened it
// (`public_catalog`). Rendered by `DeploymentRoot` before the auth gate, like
// the public auth pages, so there is no router, no shell and no organization:
// links are hash paths, the catalog is read anonymously at the deployment's
// list rates, and the one action on the page is to get an account: signup
// where the deployment offers it, sign-in otherwise.
//
// Its own bar rather than `AuthPageShell`'s: that one pins a 520px column for
// a form, and these pages are as wide as the dashboard's.
export function PublicCatalogPage({ modelId }: { modelId?: string }) {
  const { open_signup } = useDeployment()
  return (
    <div className="flex min-h-full flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4 md:px-6">
        <img
          src={`${import.meta.env.BASE_URL}favicon.svg`}
          alt=""
          className="h-6 w-[1.625rem]"
        />
        <a
          href={open_signup ? "#/signup" : "#/"}
          className={buttonVariants({ size: "sm", variant: "primary" })}
        >
          Start building with Otari
        </a>
      </header>
      <main className="mx-auto flex w-full max-w-[112.5rem] flex-col gap-6 px-4 py-5 md:px-6 md:py-6">
        <InfoBanner>
          This is the deployment's public catalog. Sign in to see the rates your
          organization is charged and to set prices.
        </InfoBanner>
        {modelId ? (
          <ModelDetailView modelId={modelId} publicView />
        ) : (
          <ModelCatalogView
            publicView
            onOpen={(id) => {
              window.location.hash = publicCatalogHref(id)
            }}
          />
        )}
      </main>
    </div>
  )
}
