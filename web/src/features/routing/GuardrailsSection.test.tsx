import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type { OrganizationContext, PolicyGuardrail } from "@/client"
import { organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

import { GuardrailsSection } from "./GuardrailsSection"

async function renderSection(
  guardrails: PolicyGuardrail[],
  context?: OrganizationContext,
) {
  const onChange = vi.fn()
  // The section asks whether the caller manages the organization. With no
  // context given, no organization answers.
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    context
      ? new Response(JSON.stringify(context), {
          headers: { "Content-Type": "application/json" },
        })
      : new Response("{}", { status: 404 }),
  )
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <GuardrailsSection
        guardrails={guardrails}
        onChange={onChange}
        isConfigured
      />
    </QueryClientProvider>,
  )
  return onChange
}

describe("GuardrailsSection", () => {
  it("says what each guardrail checks, the input alone until told otherwise", async () => {
    const onChange = await renderSection([{ profile: "pii", mode: "block" }])
    const group = screen.getByRole("group", { name: "What guardrail 1 checks" })

    expect(
      within(group).getByRole("checkbox", { name: "User input" }),
    ).toBeChecked()
    await userEvent.click(
      within(group).getByRole("checkbox", { name: "Tool results" }),
    )

    expect(onChange).toHaveBeenCalledWith([
      { profile: "pii", mode: "block", on: ["input", "tool_results"] },
    ])
  })

  it("points an organization owner or admin at the Guardrails page", async () => {
    await renderSection([], organizationContext({ role: "admin" }))
    expect(
      await screen.findByRole("link", { name: "Guardrails page" }),
    ).toHaveAttribute("href", "/tools/guardrails")
  })

  it("points nobody else at a page they may not open", async () => {
    await renderSection([], organizationContext({ role: "member" }))
    // A turn for the context read, which answers at once here, to settle.
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(screen.queryByRole("link", { name: "Guardrails page" })).toBeNull()
  })
})
