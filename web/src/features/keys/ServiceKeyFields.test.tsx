import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  NO_SERVICE_KEY,
  type ServiceKeyChoice,
  ServiceKeyFields,
  serviceKeyBody,
} from "@/features/keys/ServiceKeyFields"
import { API_ROOT } from "@/shared/api/client"
import { budget } from "@/tests/fixtures"
import { pickOption } from "@/tests/select"

const BUDGETS = [
  budget({ budget_id: "eu-ai", name: "AI" }),
  budget({ budget_id: "eu-memories", name: "Memories" }),
  budget({ budget_id: "tenant", name: "Tenant's", organization_id: "org-1" }),
]

function stubBudgets() {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const rows = String(input).startsWith(`${API_ROOT}/budgets`) ? BUDGETS : []
    return new Response(JSON.stringify(rows), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
}

function renderFields(initial: ServiceKeyChoice = NO_SERVICE_KEY) {
  const seen: ServiceKeyChoice[] = []
  function Harness() {
    const [value, setValue] = useState(initial)
    return (
      <ServiceKeyFields
        value={value}
        onChange={(next) => {
          seen.push(next)
          setValue(next)
        }}
      />
    )
  }
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <Harness />
    </QueryClientProvider>,
  )
  return { latest: () => seen.at(-1) }
}

describe("ServiceKeyFields", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("offers the deployment's budgets once the key is a service key", async () => {
    stubBudgets()
    const user = userEvent.setup()
    renderFields()

    expect(screen.queryByText("End-user budgets")).not.toBeInTheDocument()
    await user.click(screen.getByLabelText("Service key"))

    expect(await screen.findByText("AI")).toBeInTheDocument()
    expect(screen.getByText("Memories")).toBeInTheDocument()
    expect(screen.queryByText("Tenant's")).not.toBeInTheDocument()
  })

  it("picks the list and a default from it", async () => {
    stubBudgets()
    const user = userEvent.setup()
    const { latest } = renderFields({ ...NO_SERVICE_KEY, isServiceKey: true })

    await user.click(await screen.findByRole("checkbox", { name: "AI eu-ai" }))
    await user.click(
      screen.getByRole("checkbox", { name: "Memories eu-memories" }),
    )
    await pickOption(user, "Default end-user budget", "AI")

    expect(latest()).toEqual({
      isServiceKey: true,
      budgetIds: ["eu-ai", "eu-memories"],
      defaultBudgetId: "eu-ai",
    })
  })

  it("clears the default when its budget leaves the list", async () => {
    stubBudgets()
    const user = userEvent.setup()
    const { latest } = renderFields({
      isServiceKey: true,
      budgetIds: ["eu-ai", "eu-memories"],
      defaultBudgetId: "eu-ai",
    })

    await user.click(await screen.findByRole("checkbox", { name: "AI eu-ai" }))

    expect(latest()).toEqual({
      isServiceKey: true,
      budgetIds: ["eu-memories"],
      defaultBudgetId: "",
    })
  })
})

describe("serviceKeyBody", () => {
  it("sends the list and the default for a service key", () => {
    expect(
      serviceKeyBody({
        isServiceKey: true,
        budgetIds: ["eu-ai"],
        defaultBudgetId: "",
      }),
    ).toEqual({
      is_service_key: true,
      end_user_budget_ids: ["eu-ai"],
      end_user_budget_id: null,
    })
  })

  it("sends only the flag for a key that is not one, so its budgets survive", () => {
    expect(
      serviceKeyBody({
        isServiceKey: false,
        budgetIds: ["eu-ai"],
        defaultBudgetId: "eu-ai",
      }),
    ).toEqual({ is_service_key: false })
  })
})
