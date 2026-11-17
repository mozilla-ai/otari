import { useState } from "react"

import { PublicCatalogBar } from "@/features/models/PublicCatalogBar"
import { ProviderDetailView } from "@/features/providers/ProviderDetailView"
import { ProvidersView } from "@/features/providers/ProvidersView"
import {
  PROVIDERS_URL_DEFAULTS,
  providersListFromUrl,
} from "@/features/providers/providersList"
import { parseSearch } from "@/shared/helpers/search"
import { readUrlState } from "@/shared/helpers/urlState"

// The providers pages ahead of a session, where the deployment has opened its
// catalog (`public_catalog`). Rendered by `DeploymentRoot` before the auth gate,
// beside `PublicCatalogPage` and under the same bar, so there is no router and
// no organization: links are plain anchors. The list on screen is read off the
// URL once and then kept in memory, as `PublicCatalogPage` keeps its list.
export function PublicProvidersPage({ name }: { name?: string }) {
  const [list, setList] = useState(() =>
    providersListFromUrl(
      readUrlState(
        parseSearch(window.location.search.replace(/^\?/, "")),
        PROVIDERS_URL_DEFAULTS,
      ),
    ),
  )
  return (
    <div className="flex min-h-full flex-col">
      <PublicCatalogBar current={name ? null : "providers"} />
      <main className="mx-auto flex w-full max-w-[112.5rem] flex-col gap-6 px-4 py-5 md:px-6 md:py-6">
        {name ? (
          <ProviderDetailView name={name} list={list} onListChange={setList} />
        ) : (
          <ProvidersView list={list} onListChange={setList} />
        )}
      </main>
    </div>
  )
}
