import { useState } from "react"

import {
  CATALOG_URL_DEFAULTS,
  catalogListFromUrl,
} from "@/features/models/catalog"
import { ModelCatalogView } from "@/features/models/ModelCatalogPage"
import { ModelDetailView } from "@/features/models/ModelDetailPage"
import { PublicCatalogBar } from "@/features/models/PublicCatalogBar"
import { publicCatalogHref } from "@/features/models/publicCatalog"
import { parseSearch } from "@/shared/helpers/search"
import { readUrlState } from "@/shared/helpers/urlState"

// The catalog ahead of a session, where the deployment has opened it
// (`public_catalog`). Rendered by `DeploymentRoot` before the auth gate, like
// the public auth pages, so there is no router, no shell and no organization:
// links are hash paths, the catalog is read anonymously at the deployment's
// list rates, and a model's "Use this model" starts an account.
//
// Its own bar rather than `AuthPageShell`'s: that one pins a 520px column for
// a form, and these pages are as wide as the dashboard's. It carries the site's
// own navbar destinations; see `PublicCatalogBar`.
export function PublicCatalogPage({ modelId }: { modelId?: string }) {
  // A link can name the list it opens (the website's "See models" links pass
  // `?provider=`), so the first list is read off the hash. Later changes stay
  // in memory: the page is keyed on the hash, and writing one would remount it.
  const [list, setList] = useState(() =>
    catalogListFromUrl(
      readUrlState(
        parseSearch(window.location.hash.split("?")[1] ?? ""),
        CATALOG_URL_DEFAULTS,
      ),
    ),
  )
  return (
    <div className="flex min-h-full flex-col">
      <PublicCatalogBar onList={!modelId} />
      <main className="mx-auto flex w-full max-w-[112.5rem] flex-col gap-6 px-4 py-5 md:px-6 md:py-6">
        {modelId ? (
          <ModelDetailView modelId={modelId} publicView />
        ) : (
          <ModelCatalogView
            list={list}
            onListChange={setList}
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
