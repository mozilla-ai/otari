// Which of the catalog's models the Playground offers, and how they are grouped.
//
// Built from the grouped catalog (`/catalog/models`), the same read the Models
// page renders, and presented the same way: one entry per model, however many
// providers serve it, keyed by the model-level selector the gateway resolves to
// an offering itself. The flat `/models` listing would disagree with that page:
// it lists every offering separately and carries aliases and routing policies,
// which the Models page deliberately leaves to Routing. The catalog is scoped
// server-side to what the caller could route to, so nothing here re-derives
// access. What is left is editorial. The Playground only chats, so a model that
// cannot hold a conversation should not be offered, and the picker groups by
// vendor because that is how the catalog names a model.

import type { CatalogResponse } from "@/client"

export interface PlaygroundModel {
  /**
   * What to send as the `model` of a chat completion: the catalog id as a
   * selector, which the gateway resolves to the model's cheapest offering.
   */
  key: string
  /** The model's vendor, or "" where the catalog does not know one. */
  vendor: string
  /** What the picker shows: the model's catalog id. */
  label: string
}

// Models that cannot hold a conversation. Matched on the id's own slug, never
// its vendor prefix (a vendor's name says nothing about what one model does),
// because the structured metadata is empty for undiscovered offerings and the
// model's name is the one signal present on every row.
//
// It errs toward *keeping* a model, deliberately. A chat model wrongly hidden is
// a model somebody cannot use and cannot see why; an embedding model wrongly
// offered is one confusing error the first time it is picked. So the pattern
// matches only ids that are unambiguous about not being chat, and an unfamiliar
// vendor's whole catalog stays visible.
const NOT_A_CHAT_MODEL =
  /embed|whisper|\btts\b|tts-|dall-e|stable-diffusion|moderation|rerank|transcrib|\bstt\b|guard/i

/** Whether a catalog model id names something that can hold a conversation. */
export function isChatModel(modelId: string): boolean {
  const slug = modelId.slice(modelId.lastIndexOf("/") + 1)
  return !NOT_A_CHAT_MODEL.test(slug)
}

/** Split `instance:model` into its parts; a bare name has no instance. */
export function splitModelKey(key: string): {
  instance: string
  label: string
} {
  const separator = key.indexOf(":")
  if (separator === -1) return { instance: "", label: key }
  return {
    instance: key.slice(0, separator),
    label: key.slice(separator + 1),
  }
}

/**
 * The models the picker offers: one per catalog model, in catalog order.
 *
 * The key is the model-level selector, the "gateway picks the offering"
 * spelling the Models page's Use-model drawer also leads with. Until the
 * gateway has indexed the catalog that selector is null, and the first
 * offering's selector stands in so the model stays usable. The catalog arrives
 * sorted by model name and that order is preserved; duplicate keys are
 * collapsed, because two rows with one key would look like different models.
 */
export function buildPlaygroundModels(
  catalog: CatalogResponse | undefined,
): PlaygroundModel[] {
  const seen = new Set<string>()
  return (catalog?.models ?? []).flatMap((model) => {
    const key = model.selector ?? model.selectors[0] ?? ""
    if (!key || seen.has(key) || !isChatModel(model.id)) return []
    seen.add(key)
    return [{ key, vendor: model.vendor ?? "", label: model.id }]
  })
}

export interface ModelGroup {
  id: string
  label: string
  models: PlaygroundModel[]
}

/** The group heading for the pinned models, which is not a vendor. */
export const PINNED_GROUP_ID = "__pinned__"

/** The group for models whose vendor the catalog does not know. */
const UNKNOWN_VENDOR_GROUP_ID = "__other__"

/**
 * Group the picker's rows: pinned models first, then by vendor.
 *
 * A model appears once. A pinned one is in the pinned group and not also under
 * its vendor, because a picker that lists the same row twice makes the second
 * copy look like a different model.
 *
 * Vendor order is first-seen, matching `buildPlaygroundModels`, and a group
 * with nothing matching the current search is dropped rather than rendered
 * empty.
 */
export function groupPlaygroundModels(params: {
  models: readonly PlaygroundModel[]
  pinnedKeys: readonly string[]
  search: string
}): ModelGroup[] {
  const needle = params.search.trim().toLowerCase()
  const pinned = new Set(params.pinnedKeys)
  // Everything a row displays is searchable: the vendor is a group heading and
  // the key can differ from the label while the catalog is unindexed, so
  // matching the key alone would drop a model somebody is looking right at.
  const matching = params.models.filter(
    (model) =>
      !needle ||
      [model.key, model.label, model.vendor].some((value) =>
        value.toLowerCase().includes(needle),
      ),
  )
  const pinnedModels = matching.filter((model) => pinned.has(model.key))
  const byVendor = matching
    .filter((model) => !pinned.has(model.key))
    .reduce((groups, model) => {
      const bucket = groups.get(model.vendor)
      if (bucket) bucket.push(model)
      else groups.set(model.vendor, [model])
      return groups
    }, new Map<string, PlaygroundModel[]>())

  const vendorGroups = Array.from(byVendor, ([vendor, models]) => ({
    // A sentinel id, like the pinned group's, so a vendor literally named
    // "Other" cannot collide with the vendorless bucket.
    id: vendor || UNKNOWN_VENDOR_GROUP_ID,
    label: vendor || "Other",
    models,
  }))

  if (pinnedModels.length === 0) return vendorGroups
  return [
    { id: PINNED_GROUP_ID, label: "Pinned", models: pinnedModels },
    ...vendorGroups,
  ]
}

/**
 * The model to start on: the remembered one when it is still offered, else the
 * first in the catalog.
 *
 * Falling back rather than keeping a stale selection matters because a model
 * can leave the catalog between visits (a provider key revoked, a restriction
 * added), and a picker showing a model the gateway would refuse is worse than
 * one that quietly moved on.
 */
export function pickInitialModel(
  remembered: string | undefined,
  models: readonly PlaygroundModel[],
): string {
  if (remembered && models.some((model) => model.key === remembered)) {
    return remembered
  }
  return models[0]?.key ?? ""
}
