import { Dot } from "@/design-system/indicators/Dot"
import { Section } from "@/design-system/layout/Section"

/**
 * Why adding a provider is disabled: the server has no `OTARI_SECRET_KEY` to
 * encrypt a stored key with.
 *
 * A band of the page, not a tinted box: a square danger dot, the setting named
 * in mono so it is copyable by eye, and the consequence in muted prose. The
 * caller decides when it shows, from the organization context, and only on a
 * context that answered: a failed read is not evidence the key is unset.
 */
export function SecretKeyMissingBanner() {
  return (
    <Section
      className="border-y border-border py-3"
      contentClassName="flex items-start gap-3 text-sm"
    >
      <Dot className="mt-2 bg-danger" />
      <p className="text-muted">
        <span className="text-mono-caption text-foreground">
          OTARI_SECRET_KEY
        </span>{" "}
        is not set, so provider keys can&rsquo;t be encrypted at rest and adding
        providers from the dashboard is disabled. Set it on the server and
        restart to add providers here. Providers defined in{" "}
        <span className="text-mono-caption">config.yml</span> keep working
        without it.
      </p>
    </Section>
  )
}
