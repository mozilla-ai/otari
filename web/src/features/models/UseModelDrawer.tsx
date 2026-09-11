import { Button, Drawer } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { useState } from "react"

import type { CatalogModelDetail, CatalogOffering } from "@/client"
import { CopyField } from "@/design-system/actions/CopyField"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { formatRate } from "@/shared/helpers/format"
import {
  buildCurlSnippet,
  buildPythonSnippet,
  resolveSnippetBaseUrl,
} from "@/shared/helpers/requestSnippets"
import { useDeployment } from "@/shared/hooks/useDeployment"

// The way in to a model, beside the page rather than over it: a drawer from
// the right, so the offerings it talks about stay in view. Three steps: which
// provider answers (the gateway's pick, or one pinned by its selector), the
// API key, and the request to copy.

const GATEWAY_PICKS = "__gateway__"

function rate(value: number | null | undefined): string {
  return value == null ? "unpriced" : formatRate(value)
}

/** What a request sends for an offering: its short spelling where there is one. */
function selectorFor(offering: CatalogOffering): string {
  return offering.short_selector ?? offering.selector
}

export function UseModelDrawer({
  model,
  isOpen,
  onOpenChange,
  publicView,
}: {
  model: CatalogModelDetail
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** Ahead of a session: a visitor is told to sign in for a key rather than linked to the page. */
  publicView: boolean
}) {
  const deployment = useDeployment()
  const baseUrl = resolveSnippetBaseUrl(deployment)
  const [choice, setChoice] = useState(GATEWAY_PICKS)
  const [language, setLanguage] = useState("curl")

  const cheapest = model.offerings.find((o) => o.selector === model.resolves_to)
  // The gateway's own pick is the model id, where the gateway has indexed it;
  // until then the cheapest offering's selector stands in.
  const gatewayModel =
    model.selector ??
    (model.offerings[0] ? selectorFor(model.offerings[0]) : "")
  const pinned = model.offerings.find((o) => o.selector === choice)
  const sendAs = pinned ? selectorFor(pinned) : gatewayModel
  const options = [
    { value: GATEWAY_PICKS, label: "Let the gateway choose" },
    ...model.offerings.map((offering) => ({
      value: offering.selector,
      label: `${offering.provider} · ${rate(offering.pricing?.input_price_per_million)} in / ${rate(offering.pricing?.output_price_per_million)} out`,
    })),
  ]
  const input = {
    baseUrl: baseUrl ?? "",
    apiKey: "$OTARI_API_KEY",
    model: sendAs,
  }

  return (
    <Drawer isOpen={isOpen} onOpenChange={onOpenChange}>
      <Drawer.Trigger className="hidden">Open use this model</Drawer.Trigger>
      <Drawer.Backdrop className="bg-backdrop/30">
        {/* The content is the full-width lane the dialog slides in on; the
            width belongs to the dialog, or the lane shrinks and the panel
            lands at the left. */}
        <Drawer.Content placement="right">
          <Drawer.Dialog
            aria-label="Use this model"
            className="flex h-full w-[36rem] max-w-[85vw] flex-col"
          >
            <Drawer.Header className="border-b border-border px-6 py-4">
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 flex-col gap-1">
                  <Drawer.Heading className="text-heading">
                    Use {model.name}
                  </Drawer.Heading>
                  <p className="text-sm text-muted">
                    Send <code className="text-mono-caption">{sendAs}</code> as{" "}
                    <code className="text-mono-caption">model</code>.
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onPress={() => onOpenChange(false)}
                >
                  Close
                </Button>
              </div>
            </Drawer.Header>
            <Drawer.Body className="flex flex-col gap-6 overflow-y-auto px-6 py-5">
              <section className="flex flex-col gap-2">
                <h3 className="text-title">1. Which provider answers</h3>
                <p className="text-sm text-muted">
                  {model.selector ? (
                    <>
                      With the model id alone, the gateway sends the request to
                      the cheapest offering it serves
                      {cheapest ? (
                        <>
                          , today{" "}
                          <code className="text-mono-caption">
                            {selectorFor(cheapest)}
                          </code>{" "}
                          at {rate(cheapest.pricing?.input_price_per_million)}{" "}
                          per 1M in
                        </>
                      ) : null}
                      . Pick a provider to pin one instead; its selector is what
                      you send then.
                    </>
                  ) : (
                    <>
                      The gateway has not indexed this model yet, so a request
                      names a provider's selector. Pick one.
                    </>
                  )}
                </p>
                <FilterSelect
                  ariaLabel="Provider"
                  value={choice}
                  onChange={setChoice}
                  options={options}
                />
                <p className="text-caption">
                  The response's{" "}
                  <code className="text-mono-caption">model</code> is what you
                  sent; pricing, budgets and usage follow the offering that
                  answered.
                </p>
              </section>

              <section className="flex flex-col gap-2">
                <h3 className="text-title">2. Get an API key</h3>
                <p className="text-sm text-muted">
                  {publicView ? (
                    <>
                      <a href="#/" className="text-link hover:text-link-hover">
                        Sign in
                      </a>{" "}
                      and create a key on API keys, then set it as an
                      environment variable.
                    </>
                  ) : (
                    <>
                      Create a key on{" "}
                      <Link
                        to="/keys"
                        className="text-link hover:text-link-hover"
                      >
                        API keys
                      </Link>{" "}
                      and set it as an environment variable.
                    </>
                  )}
                </p>
                <CopyField
                  label="Environment"
                  value="export OTARI_API_KEY=sk-…"
                />
              </section>

              <section className="flex flex-col gap-2">
                <h3 className="text-title">3. Make your first request</h3>
                {baseUrl === undefined ? (
                  <p className="text-sm text-muted">
                    This deployment has not published its gateway address, so
                    there is no request to copy yet.
                  </p>
                ) : (
                  <>
                    <TabRow>
                      <Tab
                        isActive={language === "curl"}
                        onPress={() => setLanguage("curl")}
                      >
                        cURL
                      </Tab>
                      <Tab
                        isActive={language === "python"}
                        onPress={() => setLanguage("python")}
                      >
                        Python (OpenAI SDK)
                      </Tab>
                    </TabRow>
                    <CopyField
                      label={language === "curl" ? "cURL" : "Python"}
                      value={
                        language === "curl"
                          ? buildCurlSnippet(input)
                          : buildPythonSnippet(input)
                      }
                      multiline
                    />
                  </>
                )}
              </section>
            </Drawer.Body>
          </Drawer.Dialog>
        </Drawer.Content>
      </Drawer.Backdrop>
    </Drawer>
  )
}
