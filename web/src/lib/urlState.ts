import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";

// Keep table filter/pagination state in the URL query string, so a filtered view
// is shareable and survives the back button. Values equal to their default are
// removed to keep the URL clean, and every update is a single `setSearchParams`
// call: react-router's functional updater is based on the current location, so
// several separate calls in one tick would clobber each other rather than
// compose. `patch` therefore takes all the keys to change at once.

export function useUrlParam(key: string, defaultValue = ""): [string, (value: string) => void] {
  const [params, setParams] = useSearchParams();
  const value = params.get(key) ?? defaultValue;
  const setValue = useCallback(
    (next: string) => {
      setParams(
        (prev) => {
          const updated = new URLSearchParams(prev);
          if (!next || next === defaultValue) {
            updated.delete(key);
          } else {
            updated.set(key, next);
          }
          return updated;
        },
        { replace: true },
      );
    },
    [key, defaultValue, setParams],
  );
  return [value, setValue];
}

export interface UrlState<K extends string> {
  get: (key: K) => string;
  /** Every value of a repeatable key (`?model=a&model=b`), empty when it is absent. */
  getAll: (key: K) => string[];
  getNumber: (key: K) => number;
  /**
   * Apply several key changes in one history entry; "" or the default drops a key.
   * An array writes the key once per value, and an empty array drops it.
   */
  patch: (updates: Partial<Record<K, string | number | string[]>>) => void;
}

export function useUrlState<K extends string>(defaults: Record<K, string>): UrlState<K> {
  const [params, setParams] = useSearchParams();

  const get = useCallback((key: K) => params.get(key) ?? defaults[key], [params, defaults]);

  // Blank values are dropped so `?model=` reads as no filter, matching `get`'s
  // treatment of it (a filter on the empty string is never what a URL like that
  // means). Falls back to the key's default only when nothing is present at all.
  const getAll = useCallback(
    (key: K) => {
      const values = params.getAll(key).filter((value) => value !== "");
      if (values.length > 0) return values;
      return defaults[key] ? [defaults[key]] : [];
    },
    [params, defaults],
  );

  const getNumber = useCallback(
    (key: K) => {
      // A present but non-numeric param (e.g. a hand-edited `?size=abc`) must fall
      // back to the key's default, not 0: a 0 page size would send `limit=0` and 422.
      const parsed = Number.parseInt(params.get(key) ?? "", 10);
      if (!Number.isNaN(parsed)) {
        return parsed;
      }
      const fallback = Number.parseInt(defaults[key], 10);
      return Number.isNaN(fallback) ? 0 : fallback;
    },
    [params, defaults],
  );

  const patch = useCallback(
    (updates: Partial<Record<K, string | number | string[]>>) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [key, raw] of Object.entries(updates)) {
            if (Array.isArray(raw)) {
              // Rewritten wholesale rather than appended to: the caller passes the
              // filter's complete value set, so a removed value has to disappear.
              next.delete(key);
              for (const value of raw) {
                if (value !== "") next.append(key, value);
              }
              continue;
            }
            const value = String(raw);
            if (value === "" || value === defaults[key as K]) {
              next.delete(key);
            } else {
              next.set(key, value);
            }
          }
          return next;
        },
        { replace: true },
      );
    },
    [setParams, defaults],
  );

  return { get, getAll, getNumber, patch };
}
