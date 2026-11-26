/**
 * The bootstrap as an older deployment can actually send it, how to read one,
 * and the questions its mail state answers.
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

import type { DeploymentBootstrap, MailReadiness } from "@/client"

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
export type WireBootstrap = Partial<DeploymentBootstrap> & {
  /**
   * The field `mail` replaced.
   *
   * NOTE: Listed by hand, so nothing fails when it stops being needed. Drop it,
   * and the mapping below, once no supported deployment predates `mail`.
   */
  mail_ready?: boolean
}

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
    // On, for the reason passkeys_enabled is: a gateway old enough to omit this
    // had no way to turn organization SSO off, so reading absence as off would
    // hide the option from a deployment that serves it.
    organization_oidc_enabled: wire.organization_oidc_enabled ?? true,
    // Not "sends_only", though that is the narrower reading of the retired
    // flag. A deployment old enough to omit `mail` gated both questions on that
    // one flag, so this is the mapping under which it behaves exactly as it did
    // before `mail` existed. Reading it as the weaker state would tell everyone
    // on a delivering deployment to go and ask for a link already in their
    // inbox, which is a worse answer than the one being replaced.
    mail: wire.mail ?? (wire.mail_ready === true ? "delivers" : "none"),
    feedback_enabled: wire.feedback_enabled ?? false,
    platform_connect: wire.platform_connect ?? false,
    federation_member_rules: wire.federation_member_rules ?? false,
    // Closed by default: a gateway old enough to omit this served no catalog to
    // a visitor, and guessing open would render pages that then 401.
    public_catalog: wire.public_catalog ?? false,
    open_signup: wire.open_signup ?? false,
  }
}

/** Whether a flow that only sends a message can run on this deployment. */
export function mailCanSend(mail: MailReadiness): boolean {
  return mail !== "none"
}

/** Whether a recipient's mailbox receives what this deployment sends. */
export function mailDelivers(mail: MailReadiness): boolean {
  return mail === "delivers"
}
