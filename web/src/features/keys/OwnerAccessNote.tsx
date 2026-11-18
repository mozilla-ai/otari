/**
 * The chosen owner's model access, which a key narrows within (it can inherit
 * it or restrict to a subset, never exceed it). The operator's alone.
 */

import type { User } from "@/client"
import { accessLabel } from "@/features/models/ModelScopeControl"

export function OwnerAccessNote({
  userId,
  users,
}: {
  userId: string
  users: User[]
}) {
  const id = userId.trim()
  if (id === "") {
    return (
      <p className="text-caption">
        Choose an owner above to see the models this key can inherit.
      </p>
    )
  }
  const owner = users.find((user) => user.user_id === id)
  if (!owner) {
    return (
      <p className="text-caption">
        New user <code>{id}</code> starts unrestricted, so this key may allow
        any model.
      </p>
    )
  }
  const { text } = accessLabel(owner.allowed_models ?? undefined)
  const entries =
    owner.allowed_models && owner.allowed_models.length > 0
      ? owner.allowed_models.join(", ")
      : null
  return (
    <p className="text-caption">
      Owner <code>{id}</code> allows{" "}
      <span className="font-medium text-foreground">{text.toLowerCase()}</span>
      {entries ? (
        <>
          {" ("}
          <span className="font-mono">{entries}</span>
          {")"}
        </>
      ) : null}
      . This key inherits that, or narrows within it.
    </p>
  )
}
