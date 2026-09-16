import { useState } from "react"
import {
  FiClock,
  FiColumns,
  FiMessageSquare,
  FiSave,
  FiTrash2,
} from "react-icons/fi"
import type { PlaygroundComparison, PlaygroundConversation } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { Segmented } from "@/design-system/navigation/Segmented"
import { Popover } from "@/design-system/overlays/Popover"
import { formatDateGroup } from "@/shared/helpers/format"
import { splitModelKey } from "./helpers/playgroundModels"

interface HistoryEntry {
  id: string
  kind: "chat" | "comparison"
  title: string
  description: string
  createdAt: string
}

const PREFERENCE: Record<string, string> = {
  model_a: "A is better",
  model_b: "B is better",
  tie: "Tie",
}

export function PlaygroundHistory({
  conversations,
  comparisons,
  isOpen,
  onOpenChange,
  onLoad,
  onDeleteConversation,
  onDeleteComparison,
  onSave,
  isSaveDisabled,
  isSavePending,
  isLoading,
  error,
}: {
  conversations: readonly PlaygroundConversation[]
  comparisons: readonly PlaygroundComparison[]
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  onLoad: (id: string) => Promise<void>
  onDeleteConversation: (id: string) => Promise<void>
  onDeleteComparison: (id: string) => Promise<void>
  onSave?: () => void
  isSaveDisabled: boolean
  isSavePending: boolean
  isLoading: boolean
  error?: unknown
}) {
  const [filter, setFilter] = useState("all")
  const [search, setSearch] = useState("")
  const [isAllOpen, setIsAllOpen] = useState(false)
  // Once per render pass rather than per row, so every heading in one list is
  // measured against the same clock.
  const now = new Date()
  const [pendingDelete, setPendingDelete] = useState<HistoryEntry>()
  const [isDeleting, setIsDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<unknown>()

  const entries: HistoryEntry[] = [
    ...conversations.map((chat) => ({
      id: chat.id,
      kind: "chat" as const,
      title: chat.title,
      description: `${splitModelKey(chat.model).label} · ${chat.message_count} messages`,
      createdAt: chat.created_at,
    })),
    ...comparisons.map((comparison) => ({
      id: comparison.id,
      kind: "comparison" as const,
      title: `${splitModelKey(comparison.model_a).label} vs ${splitModelKey(comparison.model_b).label}`,
      description: `Comparison · ${PREFERENCE[comparison.preference] ?? comparison.preference} · ${comparison.user_question}`,
      createdAt: comparison.created_at,
    })),
  ].sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt))
  const matching = entries.filter(
    (entry) =>
      (filter === "all" || entry.kind === filter) &&
      `${entry.title} ${entry.description}`
        .toLowerCase()
        .includes(search.trim().toLowerCase()),
  )
  const visible = isAllOpen ? matching : matching.slice(0, 5)
  // One heading over its own list, rather than a heading inside the first row
  // of the group it names.
  const groups = visible.reduce<{ label: string; entries: HistoryEntry[] }[]>(
    (acc, entry) => {
      const label = formatDateGroup(entry.createdAt, now)
      const last = acc[acc.length - 1]
      if (last?.label === label) last.entries.push(entry)
      else acc.push({ label, entries: [entry] })
      return acc
    },
    [],
  )
  const controls = (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2">
        <span className="text-emphasis">History</span>
        <Segmented
          label="History type"
          value={filter}
          onChange={setFilter}
          options={[
            { value: "all", label: "All" },
            { value: "chat", label: "Chats" },
            { value: "comparison", label: "Comparisons" },
          ]}
        />
      </div>
      <div className="border-y border-border px-4 py-2">
        <SearchField
          label="Search history"
          placeholder="Search history"
          value={search}
          onChange={setSearch}
        />
      </div>
    </>
  )
  const list = (
    <div className="max-h-[min(28rem,55dvh)] overflow-y-auto">
      {error ? (
        <ErrorBanner error={error} />
      ) : isLoading ? (
        <p role="status" className="p-4 text-caption">
          Loading history…
        </p>
      ) : visible.length === 0 ? (
        <p className="p-4 text-caption">
          {search ? "No matching history." : "No saved history yet."}
        </p>
      ) : (
        groups.map((group) => (
          <section key={group.label}>
            <h3 className="border-t border-border px-4 py-2 text-overline">
              {group.label}
            </h3>
            <ul>
              {group.entries.map((entry) => (
                <li
                  key={`${entry.kind}:${entry.id}`}
                  className="flex items-center gap-2 border-t border-border-subtle px-4 py-2 hover:bg-surface-alt"
                >
                  {entry.kind === "chat" ? (
                    <FiMessageSquare
                      aria-hidden
                      className="size-4 shrink-0 text-muted"
                    />
                  ) : (
                    <FiColumns
                      aria-hidden
                      className="size-4 shrink-0 text-muted"
                    />
                  )}
                  {entry.kind === "chat" ? (
                    <button
                      type="button"
                      className="flex min-h-11 min-w-0 flex-1 flex-col justify-center gap-0.5 text-left"
                      onClick={async () => {
                        await onLoad(entry.id)
                        setIsAllOpen(false)
                      }}
                    >
                      <span className="w-full truncate text-emphasis">
                        {entry.title}
                      </span>
                      <span className="w-full truncate text-caption text-subtle">
                        {entry.description}
                      </span>
                    </button>
                  ) : (
                    <div className="flex min-h-11 min-w-0 flex-1 flex-col justify-center gap-0.5">
                      <span
                        className="w-full truncate text-emphasis"
                        title={entry.title}
                      >
                        {entry.title}
                      </span>
                      <span
                        className="w-full truncate text-caption text-subtle"
                        title={entry.description}
                      >
                        {entry.description}
                      </span>
                    </div>
                  )}
                  <IconButton
                    isIconOnly
                    size="sm"
                    label={`Delete ${entry.kind} "${entry.title}"`}
                    className="shrink-0"
                    onPress={() => {
                      setDeleteError(undefined)
                      setPendingDelete(entry)
                    }}
                  >
                    <FiTrash2 aria-hidden className="size-3.5" />
                  </IconButton>
                </li>
              ))}
            </ul>
          </section>
        ))
      )}
    </div>
  )
  const save = onSave ? (
    <Button
      size="sm"
      isDisabled={isSaveDisabled}
      isPending={isSavePending}
      onPress={() => {
        onOpenChange(false)
        setIsAllOpen(false)
        onSave()
      }}
    >
      <FiSave aria-hidden className="size-3.5" /> Save current chat
    </Button>
  ) : null

  return (
    <>
      <Popover
        label="History"
        padding="none"
        placement="bottom end"
        isOpen={isOpen && !isAllOpen}
        onOpenChange={onOpenChange}
        trigger={
          // Labeled rather than icon-only, which is what gives it the edge the
          // header draws: an icon-only ghost never takes one, wherever it
          // renders (globals.css).
          <Button className="min-h-11 shrink-0 md:min-h-9">
            <FiClock aria-hidden className="size-4" /> History
          </Button>
        }
      >
        <div className="flex w-[23.75rem] max-w-[calc(100vw-2rem)] flex-col">
          {controls}
          {list}
          {save ? (
            <div className="border-t border-border px-4 py-2">{save}</div>
          ) : null}
          <div className="flex items-center justify-between gap-2 border-t border-border px-4 py-2">
            <span className="text-caption">
              Only you can see saved history.
            </span>
            <Button
              size="sm"
              onPress={() => {
                onOpenChange(false)
                setIsAllOpen(true)
              }}
            >
              View all
            </Button>
          </div>
        </div>
      </Popover>
      <Dialog
        isOpen={isAllOpen}
        onOpenChange={setIsAllOpen}
        title="History"
        size="lg"
        actions={<Button onPress={() => setIsAllOpen(false)}>Close</Button>}
      >
        <DialogSection>
          {controls}
          {list}
          {save}
        </DialogSection>
      </Dialog>
      <ConfirmDialog
        isOpen={!!pendingDelete}
        onOpenChange={(open) => {
          if (!open && !isDeleting) {
            setPendingDelete(undefined)
            setDeleteError(undefined)
          }
        }}
        heading={`Delete this ${pendingDelete?.kind === "comparison" ? "comparison" : "chat"}?`}
        body={`This permanently deletes "${pendingDelete?.title ?? ""}". It cannot be undone.`}
        confirmLabel="Delete permanently"
        isPending={isDeleting}
        error={deleteError}
        onConfirm={() => {
          if (!pendingDelete) return
          setIsDeleting(true)
          const remove =
            pendingDelete.kind === "chat"
              ? onDeleteConversation
              : onDeleteComparison
          void remove(pendingDelete.id)
            .then(() => setPendingDelete(undefined))
            .catch(setDeleteError)
            .finally(() => setIsDeleting(false))
        }}
      />
    </>
  )
}
