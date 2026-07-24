import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";

// Small helpers to keep table filter/pagination state in the URL query string,
// so a filtered view is shareable and survives the back button. Values equal to
// their default are removed from the URL to keep it clean. Writes use the
// functional updater form so several params set in one tick compose instead of
// clobbering each other, and `replace: true` so filtering does not spam history.

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

export function useUrlNumberParam(key: string, defaultValue: number): [number, (value: number) => void] {
  const [raw, setRaw] = useUrlParam(key, String(defaultValue));
  const parsed = Number.parseInt(raw, 10);
  const value = Number.isNaN(parsed) ? defaultValue : parsed;
  const setValue = useCallback((next: number) => setRaw(String(next)), [setRaw]);
  return [value, setValue];
}
