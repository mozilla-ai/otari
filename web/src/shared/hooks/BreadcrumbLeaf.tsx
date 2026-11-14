import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useState,
} from "react"

/**
 * The name of the one record a detail page shows, for the shell's breadcrumb.
 *
 * The registry names the rail destination a page sits under ("MCP servers");
 * only the page knows which record it opened ("notion"), so it hands the name
 * up here and the trail ends on it.
 */
const LeafContext = createContext<{
  leaf: string
  setLeaf: (leaf: string) => void
}>({ leaf: "", setLeaf: () => {} })

export function BreadcrumbLeafProvider({ children }: { children: ReactNode }) {
  const [leaf, setLeaf] = useState("")
  return (
    <LeafContext.Provider value={{ leaf, setLeaf }}>
      {children}
    </LeafContext.Provider>
  )
}

/** The record name the open page set, or "" when it set none. */
export function useBreadcrumbLeafValue(): string {
  return useContext(LeafContext).leaf
}

/** Ends the breadcrumb on `leaf` while the calling page is mounted. */
export function useBreadcrumbLeaf(leaf: string): void {
  const { setLeaf } = useContext(LeafContext)
  useEffect(() => {
    setLeaf(leaf)
    return () => setLeaf("")
  }, [leaf, setLeaf])
}
