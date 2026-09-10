import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { SettingRow } from "./SettingRow"

describe("SettingRow", () => {
  it("names the control by its label and its config key together", () => {
    // "Backend URL" alone is not unique on a page configuring three services,
    // and a control's accessible name has to include the visible label.
    render(
      <SettingRow
        label="Backend URL"
        labelId="row-label"
        configKey="web_search_url"
        control={<input aria-labelledby="row-label row-label-key" />}
      />,
    )

    expect(
      screen.getByLabelText("Backend URL web_search_url"),
    ).toBeInTheDocument()
  })

  it("makes the label a target that focuses the control", () => {
    // The page is nothing but labelled rows, and on a phone the label sits
    // directly above its stacked field, so it is a thing people press.
    render(
      <SettingRow
        label="Backend URL"
        controlId="backend-url"
        control={<input id="backend-url" aria-label="Backend URL" />}
      />,
    )

    expect(screen.getByText("Backend URL").tagName).toBe("LABEL")
    expect(screen.getByText("Backend URL")).toHaveAttribute(
      "for",
      "backend-url",
    )
  })

  it("says nothing when a save works, so the help line never rewraps", () => {
    // A confirmation on every row of an autosaving page is a mark the reader
    // learns to ignore, and one in flow narrows the text column while it shows.
    render(
      <SettingRow
        label="Engines"
        help="Blank uses the backend's defaults."
        control={<input aria-label="Engines" />}
      />,
    )

    expect(screen.queryByRole("status")).toBeNull()
    expect(screen.getByText("Blank uses the backend's defaults.")).toBeVisible()
  })

  it("ties a refused save to the control it came from", () => {
    render(
      <SettingRow
        label="Backend URL"
        error="Must be an http or https URL."
        errorId="row-error"
        control={
          <input aria-label="Backend URL" aria-describedby="row-error" />
        }
      />,
    )

    expect(screen.getByLabelText("Backend URL")).toHaveAccessibleDescription(
      "Must be an http or https URL.",
    )
  })

  it("draws no rule of its own, because the group divides its children", () => {
    // A border here would give every seam in a group two lines.
    const { container } = render(
      <SettingRow label="Engines" control={<input aria-label="Engines" />} />,
    )

    expect(container.firstElementChild?.className).not.toMatch(/\bborder/)
  })
})
