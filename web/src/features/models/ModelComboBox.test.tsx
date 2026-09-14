import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ModelComboBox } from "@/features/models/ModelComboBox"

/**
 * The caption line under the combo box, and why it is there when it is empty.
 *
 * This control is put in a row beside a `Field` (the tier-down and pool rows on
 * the routing page), and those rows bottom-align their children. A `Field`
 * always renders a `FieldMessages` reserve, so a combo box that rendered its
 * hint only when it had one sat a caption line lower than the field next to it:
 * 19px of reserve plus the parent's 4px gap.
 *
 * jsdom does no layout, so the reserve is asserted as the class that carries it
 * rather than as a measured height, the same limit `FieldMessages.test.tsx`
 * works under.
 */
const DISCOVERABLE = {
  providers: [
    {
      provider: "openai",
      ok: true,
      models: [{ key: "openai:gpt-5-mini" }],
    },
  ],
}

const CATALOG = {
  object: "list",
  data: [
    {
      id: "openai:gpt-5-mini",
      object: "model",
      created: 0,
      owned_by: "openai",
      pricing_source: "none",
    },
    // An alias: a display name rather than a model a provider serves, so it is
    // no answer to "which selector is this rate stored under".
    {
      id: "fast",
      object: "model",
      created: 0,
      owned_by: "otari",
      pricing_source: "none",
    },
  ],
}

// Per URL, because the point of `source` is that only one of the two reads is
// ever made and a single blanket response would hide which.
function mockBothSources() {
  const urls: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    urls.push(url)
    return new Response(
      JSON.stringify(
        url.includes("/models/discoverable") ? DISCOVERABLE : CATALOG,
      ),
      { status: 200, headers: { "Content-Type": "application/json" } },
    )
  })
  return urls
}

function mockApi() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
    return new Response(JSON.stringify(DISCOVERABLE), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

// The reserve lives on the wrapper `FieldMessages` renders, which is the last
// child of the combo box root whatever the hint says. Anchored on the label
// rather than on the container, whose first child is a react-aria `<template>`,
// and structurally rather than by class, which pins a token that gets renamed.
const captionLine = (label: HTMLElement) =>
  label.parentElement?.lastElementChild as HTMLElement | null

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ModelComboBox", () => {
  it("holds the caption line open while it has nothing to say", async () => {
    mockApi()
    renderWithClient(
      <ModelComboBox label="Use instead" value="" onChange={() => {}} />,
    )
    // Past the loading hint, so the line is genuinely empty rather than holding
    // "Loading models from your providers…".
    await screen.findByRole("combobox", { name: "Use instead" })
    await vi.waitFor(() => {
      expect(
        screen.queryByText(/Loading models from your providers/),
      ).toBeNull()
    })

    const line = captionLine(screen.getByText("Use instead"))
    expect(line).not.toBeNull()
    expect(line).toHaveTextContent("")
    // The variable rather than a pixel, so a retune of the caption carries the
    // reserve with it.
    expect(line).toHaveClass("min-h-[var(--text-caption-step--line-height)]")
  })

  it("announces the hint on the input, not just beside it", async () => {
    mockApi()
    renderWithClient(
      <ModelComboBox
        label="Use instead"
        value=""
        onChange={() => {}}
        description="Pick a model or type one."
      />,
    )
    const input = await screen.findByRole("combobox", { name: "Use instead" })
    await screen.findByText("Pick a model or type one.")

    // HeroUI's `Description` is what wires the caption to the input. A bare
    // node in its place renders the same text and leaves `aria-describedby`
    // null, which is silent to a screen reader: this line is where "Could not
    // list models for openai" is said.
    const describedBy = input.getAttribute("aria-describedby")
    expect(describedBy).not.toBeNull()
    expect(document.getElementById(describedBy!)).toHaveTextContent(
      "Pick a model or type one.",
    )
  })

  it("says why the popover is empty when nothing was discovered", async () => {
    // Both issues reported this state: the chevron pointed up, so the menu was
    // open, and the box under it was empty and silent. A gateway with no
    // provider credential is the ordinary case here, not an edge.
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () =>
        new Response(JSON.stringify({ providers: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    )
    renderWithClient(
      <ModelComboBox label="Serves" value="" onChange={() => {}} />,
    )
    await screen.findByRole("combobox", { name: "Serves" })

    // The trigger, not the input: `menuTrigger="input"` means focus alone does
    // not open the list.
    await userEvent.click(screen.getByRole("button"))

    // Actionable: "no models" alone leaves an operator with nowhere to go, and
    // the sentence names the credential rather than the page that holds one,
    // since a hosted deployment keeps those under the organization.
    expect(
      await screen.findByText(/Add a provider credential/),
    ).toBeInTheDocument()
  })

  it("offers what was discovered when there is something to offer", async () => {
    mockApi()
    renderWithClient(
      <ModelComboBox label="Serves" value="" onChange={() => {}} />,
    )
    await screen.findByRole("combobox", { name: "Serves" })

    await userEvent.click(screen.getByRole("button"))

    expect(
      await screen.findByRole("option", { name: "openai:gpt-5-mini" }),
    ).toBeInTheDocument()
  })

  it("puts a hint it does have on that same line", async () => {
    mockApi()
    renderWithClient(
      <ModelComboBox
        label="Use instead"
        value=""
        onChange={() => {}}
        description="Pick a model or type one."
      />,
    )
    await screen.findByText("Pick a model or type one.")

    const line = captionLine(screen.getByText("Use instead"))
    expect(line).toHaveTextContent("Pick a model or type one.")
    expect(line).toHaveClass("min-h-[var(--text-caption-step--line-height)]")
  })

  describe("over the catalog", () => {
    it("asks the catalog and leaves discovery alone", async () => {
      // /v1/models/discoverable is a deployment-operator read, so a
      // tenant-facing form making it would paint a refusal rather than a list.
      const urls = mockBothSources()
      renderWithClient(
        <ModelComboBox
          label="Model key"
          value=""
          onChange={() => {}}
          source="catalog"
        />,
      )
      await screen.findByRole("combobox", { name: "Model key" })
      await vi.waitFor(() => {
        expect(urls.some((url) => url.endsWith("/models"))).toBe(true)
      })

      expect(urls.some((url) => url.includes("/models/discoverable"))).toBe(
        false,
      )
    })

    it("offers what the catalog serves, and not the names that only stand for it", async () => {
      mockBothSources()
      renderWithClient(
        <ModelComboBox
          label="Model key"
          value=""
          onChange={() => {}}
          source="catalog"
        />,
      )
      await screen.findByRole("combobox", { name: "Model key" })

      await userEvent.click(screen.getByRole("button"))

      expect(
        await screen.findByRole("option", { name: "openai:gpt-5-mini" }),
      ).toBeInTheDocument()
      expect(screen.queryByRole("option", { name: "fast" })).toBeNull()
    })

    it("announces a correction beside the hint rather than in place of it", async () => {
      // The hint line carries "Showing 50 of 210 matches" while a search is
      // being narrowed, which is exactly when a form is most likely to have a
      // correction to make.
      mockBothSources()
      renderWithClient(
        <ModelComboBox
          label="Model key"
          value="gpt"
          onChange={() => {}}
          source="catalog"
          description="For example openai:gpt-4o."
          isInvalid
          errorMessage="It needs the provider prefix."
        />,
      )
      const input = await screen.findByRole("combobox", { name: "Model key" })

      expect(
        await screen.findByText("It needs the provider prefix."),
      ).toBeInTheDocument()
      expect(screen.getByText("For example openai:gpt-4o.")).toBeInTheDocument()
      expect(input).toHaveAttribute("aria-invalid", "true")
    })
  })
})
