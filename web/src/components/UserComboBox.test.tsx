import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { UserComboBox } from "@/components/UserComboBox";

describe("UserComboBox", () => {
  it("caps its width so the field and dropdown trigger stay within reach", () => {
    render(<UserComboBox value="" onChange={() => {}} users={[]} />);

    // The field is bounded rather than stretching across the whole form (#328).
    const field = screen.getByRole("combobox").closest(".max-w-md");
    expect(field).toBeInTheDocument();
  });
});
