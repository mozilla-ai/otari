/**
 * The bootstrap as an older gateway can actually send it, and how to read one.
 *
 * `DeploymentBootstrap` describes the gateway this dashboard was built beside,
 * where every field is present. A gateway built before a field was added does
 * not send it, and nothing on the wire says so: the generated type promises
 * `oauth_providers: string[]`, the payload carries no `oauth_providers`, and
 * the first `.filter` on it throws (otari#806). Version skew is a normal
 * condition while developing and an ordinary one in production, where the
 * dashboard is served by whichever gateway an operator has deployed.
 *
 * So the payload is read as the possibly older shape it is and completed once,
 * here, rather than guarded at each call site that would otherwise have to
 * remember. The alternative was tried and is what #806 is about: a guard added
 * where the crash was seen leaves the same field unguarded three lines up.
 *
 * `deployment_type` and `session_type` take no default, because they say what
 * this deployment *is* and `web/AGENTS.md` is explicit that the app must not
 * guess that. Both have been on this route since it was added, so no gateway
 * that answers it at all omits one; a payload missing them is passed through as
 * it arrived, and `DeploymentRoot` falls through to the sign-in screen.
 */

import type { DeploymentBootstrap } from "@/client"

/**
 * The fields whose absence has a safe reading, which is the same reading in
 * every case: offer nothing, claim nothing, link nowhere. A gateway too old to
 * publish a field is a gateway that cannot serve what the field describes, so
 * the empty answer is not a guess about it.
 */
type SkewProne =
  | "surfaces"
  | "sign_in_methods"
  | "oauth_providers"
  | "management_url"
  | "data_plane_url"
  | "docs_url"
  | "terms_url"
  | "privacy_url"
  | "maintenance_mode"
  | "passkeys_ready"
  | "mail_ready"

/**
 * A bootstrap as received rather than as promised.
 *
 * The response-side twin of `Defaulted` in `src/client/index.ts`: that one
 * loosens a request body the dashboard may send partially, this one loosens a
 * response an older server may send partially. Both exist because the generator
 * emits one shape for a contract that has two.
 */
export type WireBootstrap = Omit<DeploymentBootstrap, SkewProne> &
  Partial<Pick<DeploymentBootstrap, SkewProne>>

/** Complete a received bootstrap, so nothing below reads an absent field. */
export function normalizeBootstrap(wire: WireBootstrap): DeploymentBootstrap {
  return {
    ...wire,
    surfaces: wire.surfaces ?? [],
    // Empty rather than `["master_key"]`, which is what a gateway old enough to
    // omit this would in fact have accepted. Naming a credential the server
    // never published is the guess this file exists to avoid, and the sign-in
    // screen already has a sentence for a deployment offering none.
    sign_in_methods: wire.sign_in_methods ?? [],
    oauth_providers: wire.oauth_providers ?? [],
    management_url: wire.management_url ?? null,
    data_plane_url: wire.data_plane_url ?? null,
    docs_url: wire.docs_url ?? null,
    terms_url: wire.terms_url ?? null,
    privacy_url: wire.privacy_url ?? null,
    maintenance_mode: wire.maintenance_mode ?? false,
    passkeys_ready: wire.passkeys_ready ?? false,
    mail_ready: wire.mail_ready ?? false,
  }
}
