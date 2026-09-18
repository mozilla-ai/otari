// Which of the catalog's models the Playground offers, and how they are grouped.
//
// The gateway's `/models` already answers the hard half: it is scoped to what
// the caller could route to, with aliases and policies resolved, so nothing here
// re-derives access. What is left is editorial. The Playground only chats, so a
// catalog entry that cannot hold a conversation should not be offered, and the
// picker groups by provider instance because that is how a model is addressed.

import type { ModelListResponse } from "@/client"

export interface PlaygroundModel {
  /** The selector to send as `model`, which is the catalog's own id. */
  key: string
  /** The provider instance the key names, or "" for a bare model name. */
  instance: string
  /** What the picker shows: the model without its instance prefix. */
  label: string
}

// Models that cannot hold a conversation. Matched on the id because the catalog
// carries no modality: every phase of it publishes `instance:model` and a
// context window, and the provider's own naming is the only signal available.
//
// It errs toward *keeping* a model, deliberately. A chat model wrongly hidden is
// a model somebody cannot use and cannot see why; an embedding model wrongly
// offered is one confusing error the first time it is picked. So the pattern
// matches only ids that are unambiguous about not being chat, and an unfamiliar
// provider's whole catalog stays visible.
const NOT_A_CHAT_MODEL =
  /embed|whisper|\btts\b|tts-|dall-e|stable-diffusion|moderation|rerank|transcrib|\bstt\b|guard/i

/** Whether a catalog id can be sent as the `model` of a chat completion. */
export function isChatModel(modelId: string): boolean {
  return !NOT_A_CHAT_MODEL.test(modelId)
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
 * The models the picker offers, in the order the catalog returned them.
 *
 * Order is preserved rather than sorted: the catalog's own phases put
 * discovered models before priced-only ones, and re-sorting here would shuffle
 * a deployment's most-used models in among every id a provider happens to
 * publish. Duplicate ids are collapsed, because a model that is both discovered
 * and priced appears once in a picker.
 */
export function buildPlaygroundModels(
  catalog: ModelListResponse | undefined,
): PlaygroundModel[] {
  const chatKeys = (catalog?.data ?? [])
    .map((entry) => entry.id)
    .filter((id) => isChatModel(id))
  return [...new Set(chatKeys)].map((key) => ({ key, ...splitModelKey(key) }))
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
