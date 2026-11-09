import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  maxEndUsersError,
  NO_SERVICE_KEY,
  type ServiceKeyChoice,
  ServiceKeyFields,
  type ServiceKeyScope,
  serviceKeyBody,
} from "@/features/keys/ServiceKeyFields"
import { API_ROOT } from "@/shared/api/client"
import { budget } from "@/tests/fixtures"
import { pickOption } from "@/tests/select"

type ServiceKeyFieldsScope = {
  scope: ServiceKeyScope
  isLocked: boolean
}

const BUDGETS = [
  budget({ budget_id: "eu-ai", name: "AI" }),
  budget({ budget_id: "eu-memories", name: "Memories" }),
  budget({ budget_id: "tenant", name: "Tenant's", organization_id: "org-1" }),
]

const ORGANIZATION_BUDGETS = [
  {
    ...budget({ budget_id: "org-free", name: "Free tier" }),
    organization_id: "org-1",
    ceiling_count: 0,
  },
]

function stubBudgets() {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    const rows = url.startsWith(`${API_ROOT}/budgets`)
      ? BUDGETS
      : url.startsWith(`${API_ROOT}/organizations/me/budgets`)
        ? { data: ORGANIZATION_BUDGETS, count: ORGANIZATION_BUDGETS.length }
        : []
    return new Response(JSON.stringify(rows), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
}

function renderFields(
  initial: ServiceKeyChoice = NO_SERVICE_KEY,
  {
    scope = "deployment",
    isLocked = false,
  }: Partial<ServiceKeyFieldsScope> = {},
) {
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
        scope={scope}
        isLocked={isLocked}
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

  it("offers an organization's key only that organization's budgets", async () => {
    stubBudgets()
    const user = userEvent.setup()
    renderFields(NO_SERVICE_KEY, { scope: "organization" })

    await user.click(screen.getByLabelText("Service key"))

    expect(await screen.findByText("Free tier")).toBeInTheDocument()
    expect(screen.queryByText("AI")).not.toBeInTheDocument()
  })

  it("keeps an organization's key a service key once it is one", () => {
    stubBudgets()
    renderFields(
      { ...NO_SERVICE_KEY, isServiceKey: true, maxEndUsers: "10000" },
      { scope: "organization", isLocked: true },
    )

    expect(screen.getByLabelText("Service key")).toBeDisabled()
  })

  it("says the budgets failed to load, rather than that there are none, and retries", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Database error" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const user = userEvent.setup()
    renderFields({ ...NO_SERVICE_KEY, isServiceKey: true })

    expect(await screen.findByRole("alert")).toBeInTheDocument()
    expect(
      screen.queryByText(/No deployment budgets yet/),
    ).not.toBeInTheDocument()

    fetchSpy.mockRestore()
    stubBudgets()
    await user.click(screen.getByRole("button", { name: "Retry" }))

    expect(await screen.findByText("AI")).toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
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
      maxEndUsers: "",
    })
  })

  it("clears the default when its budget leaves the list", async () => {
    stubBudgets()
    const user = userEvent.setup()
    const { latest } = renderFields({
      isServiceKey: true,
      budgetIds: ["eu-ai", "eu-memories"],
      defaultBudgetId: "eu-ai",
      maxEndUsers: "",
    })

    await user.click(await screen.findByRole("checkbox", { name: "AI eu-ai" }))

    expect(latest()).toEqual({
      isServiceKey: true,
      budgetIds: ["eu-memories"],
      defaultBudgetId: "",
      maxEndUsers: "",
    })
  })

  it("names no default by its label, and picks it back as an empty value", async () => {
    stubBudgets()
    const user = userEvent.setup()
    const { latest } = renderFields({
      isServiceKey: true,
      budgetIds: ["eu-ai"],
      defaultBudgetId: "eu-ai",
      maxEndUsers: "",
    })

    await pickOption(user, "Default end-user budget", /^None: uncapped/)

    expect(latest()?.defaultBudgetId).toBe("")
    expect(
      screen.getByRole("button", { name: /Default end-user budget/ }),
    ).toHaveTextContent("None: uncapped unless a request names one")
  })

  it("starts an organization's key at the default cap and refuses it empty", async () => {
    stubBudgets()
    const user = userEvent.setup()
    const { latest } = renderFields(NO_SERVICE_KEY, { scope: "organization" })

    await user.click(screen.getByLabelText("Service key"))
    const cap = await screen.findByRole("textbox", { name: /Max end users/ })
    expect(cap).toHaveValue("10000")

    await user.clear(cap)
    expect(latest()?.maxEndUsers).toBe("")
    expect(await screen.findByText("Required")).toBeInTheDocument()
  })

  it("leaves an operator's key uncapped when the cap is empty", async () => {
    stubBudgets()
    const user = userEvent.setup()
    const { latest } = renderFields()

    await user.click(screen.getByLabelText("Service key"))
    const cap = await screen.findByRole("textbox", { name: /Max end users/ })
    expect(cap).toHaveValue("")
    await user.type(cap, "2.5")

    expect(latest()?.maxEndUsers).toBe("2.5")
    expect(
      await screen.findByText("A whole number, 0 or more"),
    ).toBeInTheDocument()
  })
})

describe("maxEndUsersError", () => {
  const choice = { ...NO_SERVICE_KEY, isServiceKey: true }

  it("takes no cap only on an operator's key", () => {
    expect(maxEndUsersError(choice, "deployment")).toBeUndefined()
    expect(maxEndUsersError(choice, "organization")).toBe("Required")
  })

  it("takes a whole number, zero included", () => {
    expect(
      maxEndUsersError({ ...choice, maxEndUsers: "0" }, "organization"),
    ).toBeUndefined()
    expect(
      maxEndUsersError({ ...choice, maxEndUsers: "-1" }, "deployment"),
    ).toBe("A whole number, 0 or more")
  })
})

describe("serviceKeyBody", () => {
  it("sends the list and the default for a service key", () => {
    expect(
      serviceKeyBody({
        isServiceKey: true,
        budgetIds: ["eu-ai"],
        defaultBudgetId: "",
        maxEndUsers: "",
      }),
    ).toEqual({
      is_service_key: true,
      end_user_budget_ids: ["eu-ai"],
      end_user_budget_id: null,
      max_end_users: null,
    })
  })

  it("sends the cap as a number", () => {
    expect(
      serviceKeyBody({
        ...NO_SERVICE_KEY,
        isServiceKey: true,
        maxEndUsers: " 500 ",
      }),
    ).toMatchObject({ max_end_users: 500 })
  })

  it("sends only the flag for a key that is not one, so its budgets survive", () => {
    expect(
      serviceKeyBody({
        isServiceKey: false,
        budgetIds: ["eu-ai"],
        defaultBudgetId: "eu-ai",
        maxEndUsers: "",
      }),
    ).toEqual({ is_service_key: false })
  })
})
