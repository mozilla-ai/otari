import type { LimitTarget } from "@/features/spend/limits"

/** What the limit dialog says about who a limit lands on, as Home words it. */
export function limitDialogCopy(target: LimitTarget): {
  title: string
  description: string
} {
  return {
    title:
      target.scope === "organization"
        ? `Spend limit for all of ${target.name}`
        : `Spend limit on ${target.name}`,
    description:
      target.scope === "org_member"
        ? "Applies to everything they spend, whichever workspace or key it goes through."
        : target.scope === "api_token"
          ? "Applies to this key alone, on top of its owner’s limit."
          : target.scope === "workspace"
            ? "Applies to everyone and every key in the workspace, together."
            : "Applies to every workspace, person and key, together.",
  }
}
