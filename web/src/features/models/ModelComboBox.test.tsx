import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
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
// child of the combo box root whatever the hint says.
const captionLine = (container: HTMLElement) =>
  container.querySelector(".text-caption") as HTMLElement | null

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ModelComboBox", () => {
  it("holds the caption line open while it has nothing to say", async () => {
    mockApi()
    const { container } = renderWithClient(
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

    const line = captionLine(container)
    expect(line).not.toBeNull()
    expect(line).toHaveTextContent("")
    // The variable rather than a pixel, so a retune of the caption carries the
    // reserve with it.
    expect(line).toHaveClass("min-h-[var(--text-caption-step--line-height)]")
  })

  it("puts a hint it does have on that same line", async () => {
    mockApi()
    const { container } = renderWithClient(
      <ModelComboBox
        label="Use instead"
        value=""
        onChange={() => {}}
        description="Pick a model or type one."
      />,
    )
    await screen.findByText("Pick a model or type one.")

    const line = captionLine(container)
    expect(line).toHaveTextContent("Pick a model or type one.")
    expect(line).toHaveClass("min-h-[var(--text-caption-step--line-height)]")
  })
})
