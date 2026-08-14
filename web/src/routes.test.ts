import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL.
const ROUTES_DIR = join(process.cwd(), "src", "routes");

// A route file may export its `Route` and nothing else.
//
// That is what lets the Vite plugin's autoCodeSplitting lift each page's
// component into its own chunk (see vite.config.ts): it rewrites a route file
// into a definition half and a component half, and a second export would have to
// be resolvable from both. In practice a stray helper exported from a route file
// pulls that route's whole import graph back into the entry bundle, silently,
// and the only symptom is a fatter first load. It is also the convention the
// platform pages arriving in this repo are written to, where page components live
// in feature modules and reach their route through `getRouteApi`.
describe("route files", () => {
  const files = readdirSync(ROUTES_DIR).filter((name) => name.endsWith(".tsx"));

  it("cover the route tree", () => {
    // A guard on the guard: an empty or moved directory would leave every
    // assertion below vacuously true.
    expect(files).toContain("__root.tsx");
    expect(files.length).toBeGreaterThan(10);
  });

  it.each(files)("%s exports Route and nothing else", (file) => {
    const declarations = readFileSync(join(ROUTES_DIR, file), "utf8")
      .split("\n")
      .filter((line) => line.startsWith("export"));

    expect(declarations).toHaveLength(1);
    expect(declarations[0]).toMatch(/^export const Route = create(RootRoute|FileRoute)\(/);
  });
});
