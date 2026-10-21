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
 */

import type { DeploymentBootstrap } from "@/client"

/**
 * A bootstrap as received rather than as promised: every field optional.
 *
 * Every field's absence has a safe reading, which is the same reading in every
 * case: offer nothing, claim nothing, link nowhere. A gateway too old to
 * publish a field is a gateway that cannot serve what the field describes, so
 * the empty answer is not a guess about it.
 *
 * Derived rather than listed, because a listed one is the same bug again: the
 * next field added to the bootstrap would be absent from an older gateway,
 * absent from the list, and dereferenced unguarded, with nothing failing to say
 * so. Deriving it instead means a new field arrives here optional and
 * `normalizeBootstrap` stops compiling until it is given a default.
 */
export type WireBootstrap = Partial<DeploymentBootstrap>

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
    docs_url: wire.docs_url ?? null,
    terms_url: wire.terms_url ?? null,
    privacy_url: wire.privacy_url ?? null,
    site_url: wire.site_url ?? null,
    maintenance_mode: wire.maintenance_mode ?? false,
    // On, because a gateway old enough to omit this had no way to turn passkeys
    // off, and the page it gates already answers "not set up" when they are not
    // ready. Hiding it on a guess would take the page away from a deployment
    // that has them.
    passkeys_enabled: wire.passkeys_enabled ?? true,
    passkeys_ready: wire.passkeys_ready ?? false,
    mail_ready: wire.mail_ready ?? false,
    feedback_enabled: wire.feedback_enabled ?? false,
    // Closed by default: a gateway old enough to omit this served no catalog to
    // a visitor, and guessing open would render pages that then 401.
    public_catalog: wire.public_catalog ?? false,
    open_signup: wire.open_signup ?? false,
  }
}
