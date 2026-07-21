import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Field } from "@/components/Field";

describe("Field", () => {
  it("caps its width so inputs don't stretch across a wide form", () => {
    render(<Field label="API base" value="" onChange={() => {}} />);

    // Bounded rather than filling the whole form width (#328).
    const field = screen.getByRole("textbox").closest(".max-w-md");
    expect(field).toBeInTheDocument();
  });
});
