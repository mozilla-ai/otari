import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import type { GuardrailParameterSpec } from "@/client"
import { useGuardrailParameterForm } from "@/features/guardrails/useGuardrailParameterForm"

function spec(
  overrides: Partial<GuardrailParameterSpec> &
    Pick<GuardrailParameterSpec, "name" | "type">,
): GuardrailParameterSpec {
  return { required: false, secret: false, storable: true, ...overrides }
}

type Props = {
  specs: GuardrailParameterSpec[]
  stored: Record<string, unknown> | null
  identity: string
}

function render(initial: Props) {
  return renderHook(
    ({ specs, stored, identity }: Props) =>
      useGuardrailParameterForm(specs, stored, identity),
    { initialProps: initial },
  )
}

describe("useGuardrailParameterForm", () => {
  it("seeds from the stored kwargs", () => {
    const { result } = render({
      specs: [spec({ name: "policy", type: "string" })],
      stored: { policy: "No personal data." },
      identity: "a",
    })

    expect(result.current.values.policy).toBe("No personal data.")
  })

  it("keeps a half-typed value when a refetch hands it equal but new objects", () => {
    // The whole reason the effect depends on serialized specs: every fetch
    // builds fresh objects, and a reference dependency would re-seed and wipe
    // what the user was typing whenever anything else on the page saved.
    const { result, rerender } = render({
      specs: [spec({ name: "policy", type: "string" })],
      stored: { policy: "old" },
      identity: "a",
    })
    act(() => result.current.setValue("policy", "half-ty"))

    rerender({
      specs: [spec({ name: "policy", type: "string" })],
      stored: { policy: "old" },
      identity: "a",
    })

    expect(result.current.values.policy).toBe("half-ty")
  })

  it("re-seeds when the profile changes, even to one with the same parameters", () => {
    const specs = [spec({ name: "policy", type: "string" })]
    const { result, rerender } = render({
      specs,
      stored: null,
      identity: "a",
    })
    act(() => result.current.setValue("policy", "typed for a"))

    rerender({ specs, stored: null, identity: "b" })

    expect(result.current.values.policy).toBe("")
  })

  it("reports errors on check, not before", () => {
    const { result } = render({
      specs: [spec({ name: "policy", type: "string", required: true })],
      stored: null,
      identity: "a",
    })

    expect(result.current.issues).toEqual({})
    let ok = true
    act(() => {
      ok = result.current.check()
    })

    expect(ok).toBe(false)
    expect(result.current.issues.policy).toBeDefined()
  })

  it("builds the kwargs from the typed fields and the raw editor", () => {
    const { result } = render({
      specs: [spec({ name: "policy", type: "string" })],
      stored: { policy: "p", extra: 1 },
      identity: "a",
    })

    expect(result.current.build()).toEqual({ policy: "p", extra: 1 })
  })
})
