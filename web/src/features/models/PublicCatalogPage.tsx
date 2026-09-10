import { ModelCatalogView } from "@/features/models/ModelCatalogPage"
import { ModelDetailView } from "@/features/models/ModelDetailPage"
import { publicCatalogHref } from "@/features/models/publicCatalog"
import { InfoBanner } from "@/shared/components/feedback/InfoBanner"

// The catalog ahead of a session, where the deployment has opened it
// (`public_catalog`). Rendered by `DeploymentRoot` before the auth gate, like
// the public auth pages, so there is no router, no shell and no organization:
// links are hash paths, the catalog is read anonymously at the deployment's
// list rates, and the one action on the page is to sign in.
//
// Its own bar rather than `AuthPageShell`'s: that one pins a 520px column for
// a form, and these pages are as wide as the dashboard's.
export function PublicCatalogPage({ modelId }: { modelId?: string }) {
  return (
    <div className="flex min-h-full flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4 md:px-6">
        <img src="/favicon.svg" alt="" className="h-6 w-[26px]" />
        <a
          href="#/"
          className="inline-flex min-h-11 items-center text-sm font-medium text-link hover:text-link-hover"
        >
          Sign in
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
