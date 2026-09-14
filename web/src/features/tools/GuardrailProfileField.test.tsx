import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type { GuardrailCatalog } from "@/client"
import { GuardrailProfileField } from "@/features/tools/GuardrailProfileField"
import { selectTrigger } from "@/tests/select"

const SERVICE: GuardrailCatalog = {
  available: true,
  reason: null,
  profiles: [
    {
      profile: "house-policy",
      guardrail: "any_llm",
      model_id: null,
      parameters: [],
      parameters_known: true,
    },
  ],
}

const EMPTY: GuardrailCatalog = { available: false, reason: null, profiles: [] }

function field(
  props: Partial<Parameters<typeof GuardrailProfileField>[0]> = {},
) {
  return render(
    <GuardrailProfileField
      catalog={SERVICE}
      pending={false}
      value=""
      localNames={[]}
      onChange={vi.fn()}
      {...props}
    />,
  )
}

describe("GuardrailProfileField", () => {
  it("offers a locally defined guardrail beside the service's own profiles", async () => {
    const user = userEvent.setup()
    field({ localNames: ["prompt-injection"] })

    await user.click(selectTrigger("Guardrail profile"))
    expect(
      await screen.findByRole("option", {
        name: "prompt-injection (defined here)",
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("option", { name: "house-policy" }),
    ).toBeInTheDocument()
  })

  it("offers a picker on a deployment that runs no guardrails service", async () => {
    // The list used to be the sidecar's alone, so a gateway that defines its own
    // guardrails and runs no sidecar fell back to a free-text box.
    const user = userEvent.setup()
    field({ catalog: EMPTY, localNames: ["prompt-injection"] })

    await user.click(selectTrigger("Guardrail profile"))
    expect(
      await screen.findByRole("option", {
        name: "prompt-injection (defined here)",
      }),
    ).toBeInTheDocument()
  })

  it("says a chosen local guardrail runs in this gateway", async () => {
    field({ localNames: ["prompt-injection"], value: "prompt-injection" })

    expect(
      screen.getByText(/Runs in this gateway, from the guardrail defined here/),
    ).toBeInTheDocument()
  })

  it("still falls back to naming one by hand when nothing is listed", () => {
    field({ catalog: EMPTY, localNames: [] })

    expect(screen.getByLabelText("Guardrail profile")).toHaveAttribute(
      "type",
      "text",
    )
  })

  it("reports the name itself, without the label's tail", async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    field({ localNames: ["prompt-injection"], onChange })

    await user.click(selectTrigger("Guardrail profile"))
    await user.click(
      await screen.findByRole("option", {
        name: "prompt-injection (defined here)",
      }),
    )
    expect(onChange).toHaveBeenCalledWith("prompt-injection")
  })
})
