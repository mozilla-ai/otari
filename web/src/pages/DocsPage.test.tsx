import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DocsPage } from "@/pages/DocsPage";

describe("DocsPage", () => {
  it("renders the bundled dashboard guide, not a link to a separate docs site", () => {
    render(<DocsPage />);

    // The page chrome names it as the guide, and the guide content is rendered
    // inline from the bundled Markdown (docs/dashboard.md), so it is
    // discoverable without a docs site.
    expect(screen.getByRole("heading", { level: 1, name: "User guide" })).toBeInTheDocument();
    expect(screen.getByText(/Otari ships with a web admin dashboard for operators/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "The two-key model" })).toBeInTheDocument();
  });

  it("shows a single top-level heading, dropping the guide's duplicate title", () => {
    render(<DocsPage />);

    // The guide's own "# Admin dashboard" title is stripped so it does not
    // stack a second big heading under the page's "User guide" header.
    const h1s = screen.getAllByRole("heading", { level: 1 });
    expect(h1s).toHaveLength(1);
    expect(h1s[0]).toHaveTextContent("User guide");
  });

  it("renders GFM tables from the guide", () => {
    render(<DocsPage />);

    // The two-key model is a Markdown table; rendering it as a real <table>
    // proves remark-gfm is wired up (plain Markdown would leave it as text).
    expect(screen.getAllByRole("table").length).toBeGreaterThan(0);
  });

  it("rewrites sibling doc links to the GitHub source and opens them in a new tab", () => {
    render(<DocsPage />);

    // The guide links to sibling docs (e.g. configuration.md) that are not
    // bundled here, so a relative link cannot resolve inside the SPA. It is
    // rewritten to the rendered source on GitHub and opened in a new tab.
    const [configLink] = screen.getAllByRole("link", { name: /configuration/i });
    expect(configLink).toHaveAttribute(
      "href",
      "https://github.com/mozilla-ai/otari/blob/main/docs/configuration.md",
    );
    expect(configLink).toHaveAttribute("target", "_blank");
    expect(configLink).toHaveAttribute("rel", "noreferrer");
  });
});
