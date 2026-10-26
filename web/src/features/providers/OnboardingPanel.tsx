import type { ReactNode } from "react"

import { Button } from "@/design-system/actions/Button"
import { Dot } from "@/design-system/indicators/Dot"
import { Section } from "@/design-system/layout/Section"

function Step({
  n,
  title,
  children,
}: {
  n: number
  title: string
  children: ReactNode
}) {
  return (
    <li className="flex gap-3">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center bg-primary-subtle text-xs font-semibold text-primary-subtle-foreground">
        {n}
      </span>
      <div className="text-sm">
        <div className="font-medium text-foreground">{title}</div>
        <div className="text-muted">{children}</div>
      </div>
    </li>
  )
}

// Shown on first run (no provider configured yet). It disappears the moment a
// provider exists, so it is a nudge to the first key, not a permanent banner.
export function OnboardingPanel({
  onAddProvider,
  needsPricing,
  onEnablePricing,
  isEnabling,
  secretKeyConfigured,
}: {
  onAddProvider: () => void
  needsPricing: boolean
  onEnablePricing: () => void
  isEnabling: boolean
  secretKeyConfigured: boolean
}) {
  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <div className="flex items-start gap-3">
        <Dot className="mt-2.5 bg-accent" />
        <div>
          <h2 className="text-display-sub">Welcome to Otari</h2>
          <p className="mt-1 text-sm text-muted">
            You are signed in. Add a provider to start serving models: three
            quick steps.
          </p>
        </div>
      </div>
      <ol className="flex flex-col gap-3">
        <Step n={1} title="Add a provider">
          Enter a provider name (like <code>openai</code>) and its API key. Keys
          are encrypted at rest.
        </Step>
        <Step n={2} title="Test the connection">
          Use <strong>Test</strong> on the provider row to confirm the key works
          and see how many models it serves.
        </Step>
        <Step n={3} title="Send your first request">
          Once a provider exists, the <strong>Overview</strong> page usually
          offers a setup guide that hands you an API key and the call to make.
          Either way, point your app at <code>/v1</code> on this gateway with an
          API key (the one printed in the server logs on first start will do).
          See the{" "}
          {/* /welcome is served by the gateway itself, not by a client route, so this
                stays a plain path anchor: a router Link would resolve to /#/welcome, which
                the catch-all route sends back to the overview. It leaves the SPA, so open a
                new tab and the operator keeps the dashboard (as the guide's links do). */}
          <a
            href="/welcome"
            target="_blank"
            rel="noreferrer"
            className="font-medium text-link hover:text-link-hover"
          >
            quickstart
          </a>
          .
        </Step>
      </ol>
      {needsPricing ? (
        <p className="text-sm text-muted">
          Tip: <code>require_pricing</code> is on, so requests are rejected
          until pricing is set.{" "}
          <button
            type="button"
            className="font-medium text-link hover:text-link-hover disabled:opacity-(--disabled-opacity)"
            disabled={isEnabling}
            onClick={onEnablePricing}
          >
            Enable default pricing
          </button>{" "}
          to meter new models with public rates.
        </p>
      ) : null}
      <div>
        {/* Disabled at 0.4 rather than hidden. Blocked is not the same as
              empty: an operator whose server has no secret key has to see the
              action they are being denied, or the page reads as though adding a
              provider is not a thing this product does. */}
        <Button
          variant="primary"
          isDisabled={!secretKeyConfigured}
          onPress={onAddProvider}
        >
          Add your first provider
        </Button>
      </div>
    </Section>
  )
}
