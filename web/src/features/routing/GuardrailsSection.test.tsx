import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type { PolicyGuardrail } from "@/client"
import { renderWithRouter } from "@/tests/router"

import { GuardrailsSection } from "./GuardrailsSection"

async function renderSection(guardrails: PolicyGuardrail[]) {
  const onChange = vi.fn()
  await renderWithRouter(
    <GuardrailsSection
      guardrails={guardrails}
      onChange={onChange}
      isConfigured
    />,
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
})
