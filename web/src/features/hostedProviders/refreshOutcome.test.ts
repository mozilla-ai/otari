import { describe, expect, it } from "vitest"
import type {
  HostedCatalogProviderRefresh as CatalogProviderRefresh,
  HostedCatalogRefresh as CatalogRefresh,
  HostedCatalogPriceUpdate,
  HostedModelsRefresh,
} from "@/client"
import {
  catalogPreviewOutcome,
  catalogRefreshFailures,
  catalogRefreshOutcome,
  modelsPreviewOutcome,
  pricePreviewOutcome,
  refreshOutcome,
} from "./refreshOutcome"

const result = (patch: Partial<HostedModelsRefresh>): HostedModelsRefresh => ({
  added: [],
  repriced: [],
  count: 0,
  error: null,
  discovery_unsupported: false,
  ...patch,
})

describe("refreshOutcome", () => {
  it("says so when the provider lists nothing new", () => {
    expect(refreshOutcome(result({}))).toBe("The provider lists nothing new.")
  })

  it("counts the models a refresh offered", () => {
    expect(refreshOutcome(result({ added: ["gpt-4.1"] }))).toBe(
      "Offered 1 new model.",
    )
    expect(refreshOutcome(result({ added: ["a", "b", "c"] }))).toBe(
      "Offered 3 new models.",
    )
  })

  it("counts the default prices a refresh moved", () => {
    expect(refreshOutcome(result({ repriced: ["gpt-4o-mini"] }))).toBe(
      "Moved 1 default price to today's rate.",
    )
    expect(
      refreshOutcome(result({ added: ["a", "b"], repriced: ["c", "d"] })),
    ).toBe("Offered 2 new models. Moved 2 default prices to today's rate.")
  })

  it("words a refusal from the provider instead of a count", () => {
    expect(
      refreshOutcome(result({ error: "The key was refused.", added: ["a"] })),
    ).toBe("The key was refused.")
  })
})

const provider = (
  patch: Partial<CatalogProviderRefresh>,
): CatalogProviderRefresh => ({
  provider: "nebius",
  added: [],
  repriced: [],
  error: null,
  discovery_unsupported: false,
  credential_unreadable: false,
  source: "database",
  ...patch,
})

const sweep = (patch: Partial<CatalogRefresh>): CatalogRefresh => ({
  providers: [],
  removed: [],
  removed_count: 0,
  removed_price_rows: 0,
  removed_override_rows: 0,
  kept: [],
  ...patch,
})

describe("catalogRefreshOutcome", () => {
  it("names the model lists it dropped", () => {
    expect(catalogRefreshOutcome(sweep({ orphaned_rosters: ["openai"] }))).toBe(
      "Dropped the model lists of providers nothing configures any more: openai.",
    )
  })

  it("says so when the catalog already matches the page", () => {
    expect(catalogRefreshOutcome(sweep({}))).toBe(
      "Nothing to change: the catalog already matches this page.",
    )
  })

  it("counts what was offered across every provider", () => {
    const result = sweep({
      providers: [
        provider({ added: ["a", "b"] }),
        provider({ provider: "gemini", added: ["c"] }),
      ],
    })

    expect(catalogRefreshOutcome(result)).toBe("Offered 3 new models.")
  })

  it("counts what the purge removed", () => {
    expect(
      catalogRefreshOutcome(sweep({ removed: ["mzai:old"], removed_count: 1 })),
    ).toBe("Removed 1 model nothing here serves.")
  })

  it("reads the offer, the reprice and the purge in one sentence", () => {
    const result = sweep({
      providers: [provider({ added: ["a"], repriced: ["b", "c"] })],
      removed: ["mzai:old", "mzai:older"],
      removed_count: 2,
    })

    expect(catalogRefreshOutcome(result)).toBe(
      "Offered 1 new model. Moved 2 default prices to today's rate. " +
        "Removed 2 models nothing here serves.",
    )
  })

  it("keeps a provider's refusal out of the counts", () => {
    // A sweep whose only provider went quiet still purged, and the sentence
    // has to say that rather than reading as a sweep that found nothing.
    const result = sweep({
      providers: [provider({ error: "The key was refused." })],
      removed: ["mzai:old"],
      removed_count: 1,
    })

    expect(catalogRefreshOutcome(result)).toBe(
      "Removed 1 model nothing here serves.",
    )
  })
})

describe("catalogRefreshFailures", () => {
  it("is empty when every provider answered", () => {
    expect(
      catalogRefreshFailures(sweep({ providers: [provider({})] })),
    ).toEqual([])
  })

  it("names the provider beside its refusal, and offers to remove it", () => {
    // A key the upstream rejected means the provider serves nothing, so the
    // page can offer the removal.
    const result = sweep({
      providers: [
        provider({ error: "The key was refused." }),
        provider({ provider: "gemini" }),
      ],
    })

    expect(catalogRefreshFailures(result)).toEqual([
      {
        provider: "nebius",
        message: "nebius: The key was refused.",
        offerRemoval: true,
      },
    ])
  })

  it("does not offer removal for a provider defined in configuration", () => {
    // No delete reaches it, whatever page of the provider list it is on.
    const result = sweep({
      providers: [
        provider({ error: "The key was refused.", source: "config" }),
      ],
    })

    expect(catalogRefreshFailures(result)).toEqual([
      {
        provider: "nebius",
        message: "nebius: The key was refused.",
        offerRemoval: false,
      },
    ])
  })

  it("does not offer removal for a key this process cannot decrypt", () => {
    // The message tells the operator to re-enter the credential; offering to
    // destroy it in the same paragraph would be the opposite advice.
    const result = sweep({
      providers: [
        provider({
          error: "The stored key cannot be decrypted.",
          credential_unreadable: true,
        }),
      ],
    })

    expect(catalogRefreshFailures(result)).toEqual([
      {
        provider: "nebius",
        message: "nebius: The stored key cannot be decrypted.",
        offerRemoval: false,
      },
    ])
  })

  it("words a provider that cannot list its models without offering removal", () => {
    // Nothing is wrong with this provider: it serves requests, it just has no
    // model listing to be asked for, so offering to delete it would be wrong.
    const result = sweep({
      providers: [provider({ discovery_unsupported: true })],
    })

    expect(catalogRefreshFailures(result)).toEqual([
      {
        provider: "nebius",
        message: "nebius cannot list its models, so nothing new was offered.",
        offerRemoval: false,
      },
    ])
  })
})

describe("catalogPreviewOutcome", () => {
  it("names the model lists nothing configures any more", () => {
    expect(catalogPreviewOutcome(sweep({ orphaned_rosters: ["openai"] }))).toBe(
      "The model lists of providers nothing configures any more will be dropped: openai.",
    )
    expect(
      catalogPreviewOutcome(
        sweep({
          removed: ["openai:o3"],
          removed_count: 1,
          removed_price_rows: 1,
          orphaned_rosters: ["openai", "groq"],
        }),
      ),
    ).toBe(
      "1 model will be removed, taking 1 stored rate and 0 organization overrides with it. " +
        "The model lists of providers nothing configures any more will be dropped: openai, groq.",
    )
  })

  it("says so when there is nothing to remove", () => {
    expect(catalogPreviewOutcome(sweep({}))).toBe(
      "Nothing to remove: every priced model has something serving it.",
    )
  })

  it("counts the models, their rates and their overrides in future tense", () => {
    const result = sweep({
      removed: ["mzai:a", "mzai:b"],
      removed_count: 2,
      removed_price_rows: 3,
      removed_override_rows: 1,
    })

    expect(catalogPreviewOutcome(result)).toBe(
      "2 models will be removed, taking 3 stored rates and 1 organization override with them.",
    )
  })

  it("reads correctly for a single model with a single rate", () => {
    const result = sweep({
      removed: ["mzai:a"],
      removed_count: 1,
      removed_price_rows: 1,
      removed_override_rows: 0,
    })

    expect(catalogPreviewOutcome(result)).toBe(
      "1 model will be removed, taking 1 stored rate and 0 organization overrides with it.",
    )
  })
})

const prices = (
  patch: Partial<HostedCatalogPriceUpdate>,
): HostedCatalogPriceUpdate => ({
  fetched_at: "2026-10-21T00:00:00Z",
  added_count: 0,
  changed_count: 0,
  removed_count: 0,
  changes: [],
  changes_truncated: false,
  digest: "d",
  needs_review: false,
  review_reason: null,
  ...patch,
})

const noCatalog: CatalogRefresh = {
  providers: [],
  removed: [],
  removed_count: 0,
  removed_price_rows: 0,
  removed_override_rows: 0,
  kept: [],
}

describe("pricePreviewOutcome", () => {
  it("says a failed fetch rather than that nothing changed", () => {
    expect(
      pricePreviewOutcome({
        prices: null,
        price_error: "Unable to fetch",
        catalog: noCatalog,
      }),
    ).toBe("Unable to fetch")
  })

  it("says so when models.dev changed nothing", () => {
    expect(pricePreviewOutcome({ prices: null, catalog: noCatalog })).toBe(
      "models.dev prices are unchanged.",
    )
  })

  it("counts each kind of change, in the singular for one", () => {
    expect(
      pricePreviewOutcome({
        prices: prices({ changed_count: 1 }),
        catalog: noCatalog,
      }),
    ).toBe("models.dev has 1 changed price to apply.")
    expect(
      pricePreviewOutcome({
        prices: prices({ added_count: 3, removed_count: 2 }),
        catalog: noCatalog,
      }),
    ).toBe("models.dev has 3 new, 2 removed prices to apply.")
  })
})

describe("modelsPreviewOutcome", () => {
  it("names the providers with new models and counts them", () => {
    expect(
      modelsPreviewOutcome({
        ...noCatalog,
        providers: [
          provider({ provider: "fireworks", added: ["a", "b"] }),
          provider({ provider: "nebius", added: [] }),
          provider({ provider: "together", added: ["c"] }),
        ],
      }),
    ).toBe(
      "3 new models to offer on fireworks, together, each served once something prices it.",
    )
  })

  it("says so when no provider lists anything new", () => {
    expect(modelsPreviewOutcome(noCatalog)).toBe(
      "No provider lists a model it does not already offer.",
    )
  })
})
