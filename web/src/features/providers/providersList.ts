import { useEffect, useEffectEvent } from "react"

import { PAGE_SIZE_OPTIONS } from "@/design-system/data/TablePagination"
import type { CatalogPage } from "@/shared/api/models"
import type { UrlState } from "@/shared/helpers/urlState"

/** The page of a providers list on screen, 0-based. */
export interface ProvidersListState {
  page: number
  pageSize: number
}

export const PROVIDERS_URL_DEFAULTS = {
  page: "0",
  size: String(PAGE_SIZE_OPTIONS[0]),
}

export type ProvidersUrlKey = keyof typeof PROVIDERS_URL_DEFAULTS

/**
 * The page read off its URL, clamped: the URL is editable and the gateway
 * refuses a negative `skip` or a page size it does not offer.
 */
export function providersListFromUrl(
  url: Pick<UrlState<ProvidersUrlKey>, "getNumber">,
): ProvidersListState {
  return {
    page: Math.max(0, url.getNumber("page")),
    pageSize:
      PAGE_SIZE_OPTIONS.find((size) => size === url.getNumber("size")) ??
      PAGE_SIZE_OPTIONS[0],
  }
}

export function catalogPage({
  page,
  pageSize,
}: ProvidersListState): CatalogPage {
  return { skip: page * pageSize, limit: pageSize }
}

/**
 * Whether the list needs a pager: only once it holds more than the smallest
 * page, so a short list carries no controls, and a larger page size chosen
 * once can still be changed back.
 */
export function needsPager(total: number): boolean {
  return total > PAGE_SIZE_OPTIONS[0]
}

/** Moves a page past the end (a stale link, a shrunken list) back onto the last one. */
export function useClampedPage(
  list: ProvidersListState,
  onListChange: (next: ProvidersListState) => void,
  total: number | undefined,
  isPlaceholderData: boolean,
) {
  const clamp = useEffectEvent((lastPage: number) =>
    onListChange({ ...list, page: lastPage }),
  )
  useEffect(() => {
    if (total === undefined || isPlaceholderData) return
    const lastPage = Math.max(0, Math.ceil(total / list.pageSize) - 1)
    if (list.page > lastPage) clamp(lastPage)
  }, [total, isPlaceholderData, list.page, list.pageSize])
}
