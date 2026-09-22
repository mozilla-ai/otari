import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import type { OrganizationGuardrailDefinition } from "@/client"
import {
  DefinitionStatus,
  MandateConsequenceMark,
  ServedUncheckedBanner,
} from "@/features/guardrails/GuardrailStatus"

function definition(
  overrides: Partial<OrganizationGuardrailDefinition>,
): OrganizationGuardrailDefinition {
  return {
    id: "d1",
    organization_id: "o1",
    name: "prod-lakera",
    guardrail_name: "lakera_guard",
    enabled: true,
    build_state: "built",
    create_kwargs: {},
    create_secrets: {},
    secrets_decryptable: true,
    created_at: "2026-09-22T00:00:00Z",
    updated_at: "2026-09-22T00:00:00Z",
    ...overrides,
  }
}

describe("DefinitionStatus", () => {
  it("says a failed build is not running, in danger ink", () => {
    render(
      <DefinitionStatus definition={definition({ build_state: "failed" })} />,
    )
    expect(screen.getByText("Not running")).toHaveClass("text-danger")
  })

  it("keeps an unreadable credential apart from a failed build", () => {
    render(
      <DefinitionStatus
        definition={definition({
          build_state: "failed",
          secrets_decryptable: false,
        })}
      />,
    )
    expect(screen.getByText("Credentials unreadable")).toBeInTheDocument()
  })

  it("states a running one quietly", () => {
    render(<DefinitionStatus definition={definition({})} />)
    expect(screen.getByText("Active")).not.toHaveClass("text-danger")
  })
})

describe("MandateConsequenceMark", () => {
  it("draws nothing for a mandate that serves as configured", () => {
    const { container } = render(<MandateConsequenceMark consequence="" />)
    expect(container).toBeEmptyDOMElement()
  })

  it("names what happens to the requests", () => {
    render(<MandateConsequenceMark consequence="unchecked" />)
    expect(screen.getByText("Requests served unchecked")).toBeInTheDocument()
  })
})

describe("ServedUncheckedBanner", () => {
  it("puts the consequence first and counts mandates", () => {
    render(<ServedUncheckedBanner count={2} />)
    expect(
      screen.getByText(
        "2 mandated guardrails are not running, so the requests they cover are being served unchecked.",
      ),
    ).toBeInTheDocument()
  })

  it("is absent when nothing is served unchecked", () => {
    const { container } = render(<ServedUncheckedBanner count={0} />)
    expect(container).toBeEmptyDOMElement()
  })
})
