import { useCatalog } from "@/shared/api/models"

/** Where a model whose vendor nothing names is listed. */
export const OTHER_VENDOR = "Other"

export interface VendorGroup {
  vendor: string
  models: string[]
}

/**
 * Who makes a model a provider serves.
 *
 * The catalog's answer first. A provider that routes to others names the
 * vendor before a slash (`deepseek-ai/DeepSeek-V3` on Together), which is the
 * fallback, and a bare name nothing knows goes under "Other".
 */
export function vendorFor(
  model: string,
  vendors: ReadonlyMap<string, string>,
): string {
  const known = vendors.get(model)
  if (known) return known
  const slash = model.indexOf("/")
  return slash > 0 ? model.slice(0, slash) : OTHER_VENDOR
}

/** The models under their vendors, in name order, with "Other" last. */
export function groupModelsByVendor(
  models: readonly string[],
  vendors: ReadonlyMap<string, string>,
): VendorGroup[] {
  const groups = new Map<string, string[]>()
  for (const model of models) {
    const vendor = vendorFor(model, vendors)
    const group = groups.get(vendor)
    if (group) group.push(model)
    else groups.set(vendor, [model])
  }
  return [...groups]
    .map(([vendor, members]) => ({ vendor, models: [...members].sort() }))
    .sort((a, b) =>
      a.vendor === OTHER_VENDOR
        ? 1
        : b.vendor === OTHER_VENDOR
          ? -1
          : a.vendor.localeCompare(b.vendor),
    )
}

/**
 * Each model the catalog lists on `provider`, keyed by its bare name, with its
 * vendor. Read only while a dialog that needs it is open.
 */
export function useModelVendors(
  provider: string,
  enabled: boolean,
): ReadonlyMap<string, string> {
  const catalog = useCatalog({ provider: [provider], limit: 1000 }, enabled)
  const vendors = new Map<string, string>()
  for (const model of catalog.data?.models ?? []) {
    if (!model.vendor) continue
    for (const selector of model.selectors) {
      const prefix = `${provider}:`
      if (selector.startsWith(prefix)) {
        vendors.set(selector.slice(prefix.length), model.vendor)
      }
    }
  }
  return vendors
}
