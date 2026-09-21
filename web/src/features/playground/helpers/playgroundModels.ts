// Which of the catalog's models the Playground offers, and how they are grouped.
//
// Built from the grouped catalog (`/catalog/models`), the same read the Models
// page renders, so the two surfaces cannot disagree about what the caller may
// use. The flat `/models` listing would disagree: it also carries aliases and
// routing policies, which the Models page deliberately leaves to Routing. The
// catalog is scoped server-side to what the caller could route to, so nothing
// here re-derives access. What is left is editorial. The Playground only chats,
// so an offering that cannot hold a conversation should not be offered, and the
// picker groups by provider instance because that is how a model is addressed.

import type { CatalogResponse } from "@/client"

export interface PlaygroundModel {
  /** The offering's selector, sent as the `model` of a chat completion. */
  key: string
  /** The provider instance the key names, or "" for a bare model name. */
  instance: string
  /** What the picker shows: the model without its instance prefix. */
  label: string
}

// Models that cannot hold a conversation. Matched on the selector's model
// label, never its instance prefix (an instance named "guard" or "embeddings"
// says nothing about what its models do), because the structured metadata is
// empty for undiscovered offerings and the provider's own naming is the one
// signal present on every row.
//
// It errs toward *keeping* a model, deliberately. A chat model wrongly hidden is
// a model somebody cannot use and cannot see why; an embedding model wrongly
// offered is one confusing error the first time it is picked. So the pattern
// matches only ids that are unambiguous about not being chat, and an unfamiliar
// provider's whole catalog stays visible.
const NOT_A_CHAT_MODEL =
  /embed|whisper|\btts\b|tts-|dall-e|stable-diffusion|moderation|rerank|transcrib|\bstt\b|guard/i

/** Whether an offering selector names something that can hold a conversation. */
export function isChatModel(selector: string): boolean {
  return !NOT_A_CHAT_MODEL.test(splitModelKey(selector).label)
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
 * The models the picker offers: one row per offering, in catalog order.
 *
 * A row is an offering's selector rather than the model's catalog id, because
 * the Playground addresses one provider instance and the picker groups by it.
 * The catalog arrives sorted by model name, and preserving that order is what
 * keeps each instance group's rows name-sorted once the picker partitions
 * them; group order itself is first-seen instance. Duplicate selectors are
 * collapsed, because a selector two models somehow shared would render as two
 * rows that look like different models.
 */
export function buildPlaygroundModels(
  catalog: CatalogResponse | undefined,
): PlaygroundModel[] {
  const chatSelectors = (catalog?.models ?? [])
    .flatMap((model) => model.selectors)
    .filter((selector) => isChatModel(selector))
  return [...new Set(chatSelectors)].map((key) => ({
    key,
    ...splitModelKey(key),
  }))
}

export interface ModelGroup {
  id: string
  label: string
  models: PlaygroundModel[]
}

/** The group heading for the pinned models, which is not a provider instance. */
export const PINNED_GROUP_ID = "__pinned__"

/**
 * Group the picker's rows: pinned models first, then by provider instance.
 *
 * A model appears once. A pinned one is in the pinned group and not also under
 * its instance, because a picker that lists the same row twice makes the second
 * copy look like a different model.
 *
 * Instance order is first-seen, matching `buildPlaygroundModels`, and a group
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
  const matching = params.models.filter(
    (model) => !needle || model.key.toLowerCase().includes(needle),
  )
  const pinnedModels = matching.filter((model) => pinned.has(model.key))
  const byInstance = matching
    .filter((model) => !pinned.has(model.key))
    .reduce((groups, model) => {
      const bucket = groups.get(model.instance)
      if (bucket) bucket.push(model)
      else groups.set(model.instance, [model])
      return groups
    }, new Map<string, PlaygroundModel[]>())

  const instanceGroups = Array.from(byInstance, ([instance, models]) => ({
    id: instance || "other",
    // A bare model name belongs to no instance, which the config-driven
    // deployments produce; "Other" is honest where the instance name would be
    // an empty heading.
    label: instance || "Other",
    models,
  }))

  if (pinnedModels.length === 0) return instanceGroups
  return [
    { id: PINNED_GROUP_ID, label: "Pinned", models: pinnedModels },
    ...instanceGroups,
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
