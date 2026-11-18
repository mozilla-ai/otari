import type { ApiKey, OrganizationMember } from "@/client"

/** A person as the spend pages name them. */
export const memberName = (
  member: Pick<OrganizationMember, "full_name" | "email">,
) => member.full_name || member.email || "Someone"

/** "deploy-bot", or "Key ending 7xKp" for a key with no name. */
export const keyName = (key: Pick<ApiKey, "key_name" | "key_suffix">) =>
  key.key_name || `Key ending ${key.key_suffix ?? ""}`.trim()
