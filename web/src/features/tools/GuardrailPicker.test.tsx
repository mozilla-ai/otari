import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"

import { GuardrailPicker } from "@/features/tools/GuardrailPicker"
import { builtInGuardrail } from "@/tests/fixtures"
import { pickOption } from "@/tests/select"

const LAKERA = builtInGuardrail()
const AZURE = builtInGuardrail({
  guardrail_name: "azure_prompt_shields",
  display_name: "Azure Prompt Shields",
  vendor: "Microsoft",
})
const PROMPT_GUARD = builtInGuardrail({
  guardrail_name: "prompt_guard_2",
  display_name: "Prompt Guard 2",
  vendor: "Meta",
  backend: "local_encoder",
  requires_api_key: false,
  runnable: false,
  missing_extra: "guardrails-local",
  create_parameters: [],
})
const MODERATION = builtInGuardrail({
  guardrail_name: "openai_moderation",
  display_name: "OpenAI Moderation",
  vendor: "OpenAI",
  primary_category: "content_safety",
  categories: ["content_safety"],
})
const CATALOG = [LAKERA, AZURE, PROMPT_GUARD, MODERATION]

// The picker is controlled, so the harness holds the pair it reports and hands
// it back. Driving the real state is what proves the clearing rule below.
function Harness({
  onChange,
}: {
  onChange?: (task: string, guardrailName: string) => void
}) {
  const [task, setTask] = useState("")
  const [guardrail, setGuardrail] = useState("")
  return (
    <GuardrailPicker
      guardrails={CATALOG}
      task={task}
      guardrailName={guardrail}
      onChange={(nextTask, nextGuardrail) => {
        setTask(nextTask)
        setGuardrail(nextGuardrail)
        onChange?.(nextTask, nextGuardrail)
      }}
    />
  )
}

describe("GuardrailPicker", () => {
  it("keeps the guardrail control shut until a task is chosen, and says why", () => {
    render(<Harness />)
    expect(
      screen.getByRole("combobox", { name: /Which guardrail/ }),
    ).toBeDisabled()
    expect(
      screen.getByText("Choose what you want checked first."),
    ).toBeInTheDocument()
  })

  it("says what the chosen task catches and how many guardrails do it", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await pickOption(user, "What do you want checked?", "Prompt injection")
    expect(
      await screen.findByText(
        "Catches an attempt to override your instructions. 3 guardrails can do this.",
      ),
    ).toBeInTheDocument()
  })

  it("offers only that task's guardrails, what can run first", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await pickOption(user, "What do you want checked?", "Prompt injection")
    await user.click(screen.getByRole("combobox", { name: /Which guardrail/ }))

    const offered = (await screen.findAllByRole("option")).map(
      (option) => option.textContent,
    )
    expect(offered).toHaveLength(3)
    expect(offered[0]).toContain("Azure Prompt Shields")
    expect(offered[1]).toContain("Lakera Guard")
    expect(offered[2]).toContain("Prompt Guard 2")
    expect(offered.join(" ")).not.toContain("OpenAI Moderation")
  })

  it("dims a guardrail whose packages are missing and names the extra", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await pickOption(user, "What do you want checked?", "Prompt injection")
    await user.click(screen.getByRole("combobox", { name: /Which guardrail/ }))

    const local = await screen.findByRole("option", {
      name: /Prompt Guard 2/,
    })
    expect(local).toHaveAttribute("aria-disabled", "true")
    expect(local.textContent).toContain("install guardrails-local to use")
  })

  it("reports the guardrail it was given", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)
    await pickOption(user, "What do you want checked?", "Prompt injection")
    await user.click(screen.getByRole("combobox", { name: /Which guardrail/ }))
    await user.click(
      await screen.findByRole("option", { name: /Lakera Guard/ }),
    )

    expect(onChange).toHaveBeenLastCalledWith(
      "prompt_injection",
      "lakera_guard",
    )
  })

  it("drops the guardrail when the task changes under it", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)
    await pickOption(user, "What do you want checked?", "Prompt injection")
    await user.click(screen.getByRole("combobox", { name: /Which guardrail/ }))
    await user.click(
      await screen.findByRole("option", { name: /Lakera Guard/ }),
    )

    // Lakera does not do content safety, so leaving it selected would submit a
    // guardrail the operator can no longer see in the list.
    await pickOption(user, "What do you want checked?", "Harmful content")
    expect(onChange).toHaveBeenLastCalledWith("content_safety", "")
  })

  it("filters the list by the vendor as well as the name", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await pickOption(user, "What do you want checked?", "Prompt injection")
    const field = screen.getByRole("combobox", { name: /Which guardrail/ })
    await user.click(field)
    await user.type(field, "microsoft")

    const offered = await screen.findAllByRole("option")
    expect(offered).toHaveLength(1)
    expect(offered[0].textContent).toContain("Azure Prompt Shields")
  })

  it("says the build ships nothing rather than drawing two dead controls", () => {
    render(
      <GuardrailPicker
        guardrails={[]}
        task=""
        guardrailName=""
        onChange={vi.fn()}
      />,
    )
    expect(
      screen.getByText(/This build ships no guardrails it can run itself/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("combobox", { name: /Which guardrail/ }),
    ).not.toBeInTheDocument()
  })
})
