import { UsagePage } from "@/features/usage/UsagePage"

// The organization rail's Usage destination (otari-ai#1963): the same analytics
// as the workspace rail's, pinned to the organization-wide scope. All the
// behavior lives on UsagePage, keyed off the one prop.
export function OrganizationUsagePage() {
  return <UsagePage scope="organization" />
}
