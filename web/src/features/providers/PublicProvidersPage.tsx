import { PublicCatalogBar } from "@/features/models/PublicCatalogBar"
import { ProviderDetailView } from "@/features/providers/ProviderDetailPage"
import { ProvidersView } from "@/features/providers/ProvidersPage"

// The providers pages ahead of a session, where the deployment has opened its
// catalog (`public_catalog`). Rendered by `DeploymentRoot` before the auth gate,
// beside `PublicCatalogPage` and under the same bar, so there is no router and
// no organization: links are hash paths.
export function PublicProvidersPage({ name }: { name?: string }) {
  return (
    <div className="flex min-h-full flex-col">
      <PublicCatalogBar current={name ? null : "providers"} />
      <main className="mx-auto flex w-full max-w-[112.5rem] flex-col gap-6 px-4 py-5 md:px-6 md:py-6">
        {name ? (
          <ProviderDetailView name={name} publicView />
        ) : (
          <ProvidersView publicView />
        )}
      </main>
    </div>
  )
}
