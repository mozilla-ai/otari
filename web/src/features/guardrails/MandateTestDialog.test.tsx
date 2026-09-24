import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { GuardrailCatalog } from "@/client"
import { MandateTestDialog } from "@/features/guardrails/MandateTestDialog"
import { organizationGuardrail } from "@/tests/fixtures"

function answer(status: number, body: unknown) {
  const calls: { url: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    calls.push({
      url: String(input),
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    })
    return Response.json(body, { status })
  })
  return calls
}

function renderDialog(catalog?: GuardrailCatalog) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MandateTestDialog
        isOpen
        onClose={() => {}}
        mandate={organizationGuardrail({
          id: "m1",
          profile: "pii",
          validate_kwargs: { threshold: 0.8, api_key: "***" },
        })}
        catalog={catalog}
      />
    </QueryClientProvider>,
  )
}

function dialog() {
  return within(screen.getByRole("dialog"))
}

async function run(text: string) {
  const user = userEvent.setup()
  await user.type(dialog().getByLabelText("Text to check"), text)
  await user.click(dialog().getByRole("button", { name: "Run test" }))
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("MandateTestDialog", () => {
  it("sends the text with the stored arguments, masks included", async () => {
    const calls = answer(200, {
      valid: false,
      explanation: "email address",
      score: 0.9,
    })
    renderDialog()

    await run("mail me at a@b.c")

    const result = await screen.findByRole("status", { name: "Test result" })
    expect(result).toHaveTextContent("Flagged")
    expect(result).toHaveTextContent("email address")
    expect(calls[0]?.url).toMatch(/\/organizations\/me\/guardrails\/m1\/test$/)
    // The server swaps a *** back for what it stores.
    expect(calls[0]?.body).toEqual({
      text: "mail me at a@b.c",
      validate_kwargs: { threshold: 0.8, api_key: "***" },
    })
  })

  it("says when the service gave no verdict", async () => {
    answer(200, { valid: null, explanation: null, score: null })
    renderDialog()

    await run("hello")

    expect(
      await screen.findByRole("status", { name: "Test result" }),
    ).toHaveTextContent("No verdict")
  })

  it("says the service could not be reached where the gateway's answer is generic", async () => {
    answer(502, { detail: "Internal server error" })
    renderDialog()

    await run("hello")

    expect(
      await screen.findByText(/the guardrails service could not be reached/),
    ).toBeInTheDocument()
  })

  it("passes on why there is nothing to test against", async () => {
    answer(409, { detail: "This guardrail names no endpoint." })
    renderDialog()

    await run("hello")

    await waitFor(() =>
      expect(
        screen.getByText("This guardrail names no endpoint."),
      ).toBeInTheDocument(),
    )
  })
})
