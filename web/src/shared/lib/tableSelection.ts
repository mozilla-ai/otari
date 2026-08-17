import { useCallback, useState } from "react";
import type { Selection } from "react-aria-components";

// Row selection for the shared tables. react-aria owns the per-page selection
// (a Set of row keys, or the sentinel "all" meaning every selectable row on the
// page). On top of that we track `allMatching`: the operator's intent to act on
// every row matching the current filter, not just the page, so a bulk op can be
// set-scoped (by filter) instead of id-scoped. Any manual selection change drops
// `allMatching`, so the two never drift.

export interface TableSelection {
  selectedKeys: Selection;
  onSelectionChange: (keys: Selection) => void;
  allMatching: boolean;
  enableAllMatching: () => void;
  clear: () => void;
}

export function useTableSelection(): TableSelection {
  const [selectedKeys, setSelectedKeys] = useState<Selection>(() => new Set<string>());
  const [allMatching, setAllMatching] = useState(false);

  const onSelectionChange = useCallback((keys: Selection) => {
    setSelectedKeys(keys);
    setAllMatching(false);
  }, []);

  const clear = useCallback(() => {
    setSelectedKeys(new Set<string>());
    setAllMatching(false);
  }, []);

  const enableAllMatching = useCallback(() => setAllMatching(true), []);

  return { selectedKeys, onSelectionChange, allMatching, enableAllMatching, clear };
}

/**
 * The selected row ids on the current page. `selectableKeys` is the page's list
 * of selectable row keys, in order, used both to expand the "all" sentinel and
 * to keep the result ordered and free of stale/disabled keys.
 */
export function resolveSelectedIds(selection: Selection, selectableKeys: string[]): string[] {
  if (selection === "all") {
    return selectableKeys;
  }
  return selectableKeys.filter((key) => selection.has(key));
}

/** How many rows the page selection covers (expanding the "all" sentinel). */
export function selectionCount(selection: Selection, selectableKeys: string[]): number {
  return resolveSelectedIds(selection, selectableKeys).length;
}
