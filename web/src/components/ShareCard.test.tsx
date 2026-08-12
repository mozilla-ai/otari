import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { CardStat } from "@/lib/shareCard";

import { CARD_SIZES, ShareCard, truncateModel } from "./ShareCard";

function renderCard(overrides: Partial<Parameters<typeof ShareCard>[0]> = {}) {
  return render(
    <ShareCard
      ratio="square"
      theme="dark"
      title="Where my tokens went"
      scope="Aug 1 – Aug 7, 2026 · all models"
      hero={{ id: "requests", label: "Requests", value: "1,204" }}
      models={[]}
      stats={[]}
      {...overrides}
    />,
  );
}

describe("ShareCard", () => {
  it("renders the hero and its label", () => {
    renderCard();
    expect(screen.getByText("1,204")).toBeInTheDocument();
    expect(screen.getByText(/Requests/)).toBeInTheDocument();
  });

  it("shows an explicit empty state instead of a hole when there is no hero", () => {
    renderCard({ hero: undefined });
    expect(screen.getByText("No usage in this range")).toBeInTheDocument();
  });

  it("marks a caveated stat so an unpriced dollar figure is never published bare", () => {
    renderCard({
      hero: { id: "cost", label: "Spend", value: "$412.00", caveated: true } satisfies CardStat,
    });
    expect(screen.getByText("Spend*")).toBeInTheDocument();
  });

  it("prints a legend for the caveat asterisk, which travels with the file", () => {
    renderCard({
      hero: { id: "cost", label: "Spend", value: "$412.00", caveated: true },
      unpricedRequests: 63,
    });
    expect(screen.getByText(/63 requests unpriced/)).toBeInTheDocument();
  });

  it("does not explain a mark the card is not showing", () => {
    renderCard({ hero: { id: "requests", label: "Requests", value: "7" }, unpricedRequests: 63 });
    expect(screen.queryByText(/unpriced/)).not.toBeInTheDocument();
  });

  it("hardcodes the otari URL rather than deriving it from the gateway's own host", () => {
    renderCard();
    expect(screen.getByText("otari.ai")).toBeInTheDocument();
  });

  it("carries the Otari mark inline, not as an external reference", () => {
    const { container } = renderCard();
    // An <img src="/favicon.svg"> would render as nothing once the card is
    // rasterized: that document cannot fetch anything external.
    const mark = container.querySelector('svg[viewBox="0 0 272 250"]');
    expect(mark).not.toBeNull();
    expect(mark?.querySelector("path")).not.toBeNull();
    expect(container.querySelector("img")).toBeNull();
  });

  it("names the scope, so the denominator behind the claim is never ambiguous", () => {
    renderCard({ scope: "Aug 1 – Aug 7, 2026 · user: ana" });
    expect(screen.getByText("Aug 1 – Aug 7, 2026 · user: ana")).toBeInTheDocument();
  });

  it("renders both ratios at their exact pixel sizes", () => {
    const { rerender } = renderCard();
    expect(screen.getByRole("img", { name: /Usage card/ })).toHaveStyle({ width: `${CARD_SIZES.square.width}px` });
    rerender(
      <ShareCard
        ratio="landscape"
        theme="light"
        title="t"
        scope="s"
        hero={undefined}
        models={[]}
        stats={[]}
      />,
    );
    expect(screen.getByRole("img", { name: /Usage card/ })).toHaveStyle({ height: `${CARD_SIZES.landscape.height}px` });
  });

  it("never renders type below the feed-legibility floor", () => {
    renderCard({
      models: [{ key: "llama-3.3-70b", label: "llama-3.3-70b", tokens: 700, cost: 0, requests: 3, isOther: false }],
      stats: [{ id: "requests", label: "Requests", value: "1,204" }],
    });
    for (const node of document.querySelectorAll<HTMLElement>("[style*='font-size']")) {
      const size = Number.parseFloat(node.style.fontSize);
      if (!Number.isNaN(size)) {
        expect(size).toBeGreaterThanOrEqual(28);
      }
    }
  });
});

describe("ShareCard height budget", () => {
  const models = (n: number) =>
    Array.from({ length: n }, (_, i) => ({
      key: `m-${i}`,
      label: `model-${i}`,
      tokens: 1000 - i,
      cost: 0,
      requests: 1,
      isOther: false,
    }));

  // A fixed frame plus fixed row heights broke both ways: three rows left a third
  // of a square card blank, and nine overflowed so flex-shrink collapsed the title
  // to zero height. Rows divide a budget instead.
  it.each([1, 3, 5, 9])("keeps rows legible and the hero present at %i rows", (n) => {
    const { container } = renderCard({ models: models(n), stats: [{ id: "requests", label: "Requests", value: "7" }] });
    // Assert the count first. An earlier version filtered heights to `> 30`
    // before asserting, which discarded exactly the collapsed row it was meant to
    // catch and passed over an empty list.
    const rows = Array.from(container.querySelectorAll<HTMLElement>("[data-share-row]"));
    expect(rows).toHaveLength(n);
    for (const row of rows) {
      expect(Number.parseFloat(row.style.height)).toBeGreaterThanOrEqual(34);
    }
    expect(screen.getByText("1,204")).toBeInTheDocument();
  });

  it("gives the hero less height when the list is long, since nothing else can yield", () => {
    const short = renderCard({ models: models(3) });
    const shortHero = short.container.querySelector<HTMLElement>('[style*="font-size: 200px"]');
    expect(shortHero).not.toBeNull();
    short.unmount();

    const long = renderCard({ models: models(9) });
    expect(long.container.querySelector('[style*="font-size: 200px"]')).toBeNull();
    expect(long.container.querySelector('[style*="font-size: 150px"]')).not.toBeNull();
  });
});

describe("truncateModel", () => {
  it("truncates in the middle, because a model id is identified by its tail", () => {
    const truncated = truncateModel("claude-sonnet-4-5-20260514", 20);
    expect(truncated).toContain("…");
    expect(truncated.endsWith("20260514")).toBe(true);
    expect(truncated.length).toBeLessThanOrEqual(20);
  });

  it("leaves a short name alone", () => {
    expect(truncateModel("gpt-4o")).toBe("gpt-4o");
  });
});
