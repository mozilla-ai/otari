// How the dashboard's URLs encode their query string.
//
// Every search param the pages read is a flat string, and a repeated key
// (`?model=a&model=b`) is that filter's whole value set. TanStack Router's
// default codec JSON-encodes values instead, which would rewrite that as
// `?model=["a","b"]`. These URLs are hand-written, shared and bookmarked (the
// e2e suite opens them directly, and Usage links into Activity with repeated
// params), so the router is handed this codec and every existing link keeps
// working unchanged.

export type DashboardSearch = Record<string, string | string[] | undefined>;

export function parseSearch(searchStr: string): DashboardSearch {
  const params = new URLSearchParams(searchStr);
  // Built through a Map so a `?__proto__=x` param lands as an ordinary key:
  // Object.fromEntries defines own properties, where `object[key] = value`
  // would hit the prototype setter instead.
  const entries = new Map<string, string | string[]>();
  for (const key of params.keys()) {
    if (entries.has(key)) continue;
    const values = params.getAll(key);
    entries.set(key, values.length > 1 ? values : values[0]);
  }
  return Object.fromEntries(entries);
}

export function stringifySearch(search: DashboardSearch): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(search)) {
    if (value === undefined) continue;
    for (const one of Array.isArray(value) ? value : [value]) {
      params.append(key, String(one));
    }
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}

/**
 * Search validation for the root route, which every route inherits.
 *
 * The filters are per-page and untyped on the wire (Activity alone reads a
 * dozen), so the shared contract is "strings, or sets of them" rather than a
 * schema per route. It is what lets a link name any param the destination reads
 * without the route restating it.
 */
export function validateSearch(search: Record<string, unknown>): DashboardSearch {
  return search as DashboardSearch;
}
