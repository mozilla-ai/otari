import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { SettingsSection } from "./SettingsSection"

describe("SettingsSection", () => {
  it("renders the title as a level-two heading", () => {
    render(
      <SettingsSection title="Provider keys">
        <span>Body</span>
      </SettingsSection>,
    )

    expect(
      screen.getByRole("heading", { level: 2, name: "Provider keys" }),
    ).toBeInTheDocument()
  })

  it("renders the description when provided", () => {
    render(
      <SettingsSection
        title="Provider keys"
        description="Keys served by the platform."
      >
        <span>Body</span>
      </SettingsSection>,
    )

    expect(screen.getByText("Keys served by the platform.")).toBeInTheDocument()
  })

  it("renders right-aligned actions when provided", () => {
    render(
      <SettingsSection
        title="Provider keys"
        actions={<button type="button">Add key</button>}
      >
        <span>Body</span>
      </SettingsSection>,
    )

    expect(screen.getByRole("button", { name: "Add key" })).toBeInTheDocument()
  })

  it("renders its children as the section body", () => {
    render(
      <SettingsSection title="Provider keys">
        <span>Section body content</span>
      </SettingsSection>,
    )

    expect(screen.getByText("Section body content")).toBeInTheDocument()
  })
})
