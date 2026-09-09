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
 * this deployment *is*, and `web/AGENTS.md` refuses to guess that for a
 * bootstrap that never arrived. The same refusal is the right one for a
 * bootstrap that arrived without them, though that file states only the first
 * case. Both have been on this route since it was added (`88c24ac13`), so no
 * gateway that answers it at all omits one; a payload missing them is passed
 * through as it came, and `DeploymentRoot` falls through to the sign-in screen,
 * where `Login` says the gateway published no way in.
 *
 * That last part is a claim about this gateway only. `bootstrap.py` says the
 * contract is shared with otari.ai, which serves the same shape from its own
 * codebase, so the cast in `normalizeBootstrap`'s return type is guarded by
 * history rather than by the compiler.
 */

import type { Defaulted, DeploymentBootstrap } from "@/client"

/**
 * The fields whose absence has a safe reading, which is the same reading in
 * every case: offer nothing, claim nothing, link nowhere. A gateway too old to
 * publish a field is a gateway that cannot serve what the field describes, so
 * the empty answer is not a guess about it.
 *
 * Derived rather than listed, because a listed one is the same bug again: the
 * next field added to the bootstrap would be absent from an older gateway,
 * absent from the list, and dereferenced unguarded, with nothing failing to say
 * so. Subtracting instead means a new field arrives in `WireBootstrap` optional
 * and `normalizeBootstrap` stops compiling until it is given a default.
 */
type SkewProne = Exclude<
  keyof DeploymentBootstrap,
  "deployment_type" | "session_type"
>

/**
 * A bootstrap as received rather than as promised.
 *
 * `Defaulted` itself, applied to a response rather than to a request body: that
 * use loosens what the dashboard may send partially, this one what an older
 * gateway may answer partially. Both exist because the generator emits one
 * shape for a contract that has two.
 */
export type WireBootstrap = Defaulted<DeploymentBootstrap, SkewProne>

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
