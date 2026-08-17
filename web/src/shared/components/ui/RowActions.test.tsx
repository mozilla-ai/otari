import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { RowActions } from "./RowActions"

describe("RowActions", () => {
  it("renders its children inside a single row container", () => {
    render(
      <RowActions>
        <button type="button">Edit</button>
        <button type="button">Delete</button>
      </RowActions>,
    )

    const editButton = screen.getByRole("button", { name: "Edit" })
    const deleteButton = screen.getByRole("button", { name: "Delete" })
    expect(editButton).toBeInTheDocument()
    expect(deleteButton).toBeInTheDocument()
    expect(editButton.parentElement).toBe(deleteButton.parentElement)
  })
})
