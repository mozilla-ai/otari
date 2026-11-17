import { BASE_PATH } from "@/shared/helpers/appLocation"

/**
 * A link the server built into this dashboard, made absolute so it can be
 * pasted into a chat or an email.
 *
 * The server can only make it absolute when it knows its public address; without
 * one it answers the dashboard route alone (`/accept-invitation#token=…`). That
 * is resolved against this page's origin and the base the dashboard is served
 * under, which the server does not know.
 */
export function absoluteDashboardLink(
  link: string,
  origin: string = window.location.origin,
): string {
  if (/^https?:\/\//i.test(link)) return link
  return `${origin}${BASE_PATH}${link}`
}
