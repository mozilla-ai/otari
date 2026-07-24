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
  getNumber: (key: K) => number;
  /** Apply several key changes in one history entry; "" or the default drops a key. */
  patch: (updates: Partial<Record<K, string | number>>) => void;
}

export function useUrlState<K extends string>(defaults: Record<K, string>): UrlState<K> {
  const [params, setParams] = useSearchParams();

  const get = useCallback((key: K) => params.get(key) ?? defaults[key], [params, defaults]);

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
    (updates: Partial<Record<K, string | number>>) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [key, raw] of Object.entries(updates)) {
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

  return { get, getNumber, patch };
}
