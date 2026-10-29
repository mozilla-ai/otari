import { useState } from "react"

import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import { AccessChooser } from "@/features/tools/AccessChooser"
import {
  type AccessMode,
  accessDraft,
  isDraftComplete,
  toolsFor,
} from "@/features/tools/mcpAccess"
import { filterUsers, type GrantUser } from "@/features/tools/mcpGrantSubjects"

/** What a submit sends: the keys and users picked, and the tools they get. */
export interface GrantDraft {
  api_key_ids: string[]
  user_ids: string[]
  tools: string[]
}

/**
 * Adds exceptions for several keys and users at once, or changes one.
 *
 * Adding picks from the workspace's users, each with the keys of theirs the
 * page can see beneath them, so a user and one of their keys can be granted
 * together; the key's own rule then wins for that key. Editing names its one
 * subject and offers only the access.
 */
export function McpGrantDialog({
  isOpen,
  onOpenChange,
  users,
  editing,
  tools,
  isPending,
  error,
  onSubmit,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** Who can be picked. Ignored when editing. */
  users: readonly GrantUser[]
  /** The grant being changed, by who it is for and what it allows now. */
  editing?: { name: string; tools: string[] }
  /** What the server's last check listed, to tick from. */
  tools: readonly string[]
  isPending: boolean
  error: unknown
  onSubmit: (draft: GrantDraft) => void
}) {
  const seed = accessDraft(editing?.tools ?? ["*"])
  const [mode, setMode] = useState<AccessMode>(seed.mode)
  const [picked, setPicked] = useState<string[]>(seed.picked)
  const [query, setQuery] = useState("")
  const [keyIds, setKeyIds] = useState<readonly string[]>([])
  const [userIds, setUserIds] = useState<readonly string[]>([])

  const toggle = (
    list: readonly string[],
    set: (next: readonly string[]) => void,
    id: string,
    isSelected: boolean,
  ) => set(isSelected ? [...list, id] : list.filter((entry) => entry !== id))
  const hasSubject = editing !== undefined || keyIds.length + userIds.length > 0
  const isComplete = hasSubject && isDraftComplete(mode, picked)
  const shown = filterUsers(users, query)
  const picks = keyIds.length + userIds.length

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title={editing ? `Edit ${editing.name}` : "Add rule"}
      description={
        editing
          ? undefined
          : "Pick the sources: single keys, or users, which covers every key they own. Each gets this rule; one that already has a rule gets this one instead."
      }
      submitLabel={editing ? "Save rule" : "Add rule"}
      onSubmit={() => {
        if (!isComplete) return
        onSubmit({
          api_key_ids: [...keyIds],
          user_ids: [...userIds],
          tools: toolsFor(mode, picked),
        })
      }}
      isPending={isPending}
      isSubmitDisabled={!isComplete}
      isDirty={
        picks > 0 || mode !== seed.mode || picked.join() !== seed.picked.join()
      }
      error={error}
    >
      {editing ? null : (
        <div className="flex flex-col gap-2">
          <SearchField
            label="Source"
            placeholder="Name or key"
            value={query}
            onChange={setQuery}
          />
          <p className="text-caption" aria-live="polite">
            {picks === 0 ? "Nothing picked yet." : `${picks} picked.`}
          </p>
          {shown.length === 0 ? (
            <p className="text-caption">
              {users.length === 0
                ? "This workspace has no keys or members to pick."
                : "Nothing matches."}
            </p>
          ) : (
            <ul
              aria-label="Keys and users"
              className="flex max-h-72 flex-col gap-3 overflow-y-auto border border-border p-3"
            >
              {shown.map((user) => (
                <li key={user.userId} className="flex min-w-0 flex-col gap-1">
                  <Checkbox
                    isSelected={userIds.includes(user.userId)}
                    onChange={(next) =>
                      toggle(userIds, setUserIds, user.userId, next)
                    }
                    ariaLabel={`${user.label}, all their keys`}
                    className="min-h-11 md:min-h-0"
                  >
                    <span className="min-w-0 break-words">
                      <span className="font-medium">{user.label}</span>
                      <span className="text-caption"> · all their keys</span>
                    </span>
                  </Checkbox>
                  {user.keys.length > 0 ? (
                    <ul className="flex flex-col gap-1 pl-6">
                      {user.keys.map((key) => (
                        <li key={key.id} className="min-w-0">
                          <Checkbox
                            isSelected={keyIds.includes(key.id)}
                            onChange={(next) =>
                              toggle(keyIds, setKeyIds, key.id, next)
                            }
                            ariaLabel={`Key ${key.name} of ${user.label}`}
                            className="min-h-11 md:min-h-0"
                          >
                            <span className="flex min-w-0 flex-wrap items-center gap-2">
                              <span className="font-mono text-sm break-all">
                                {key.name}
                              </span>
                              {key.isActive ? null : <Chip>inactive</Chip>}
                            </span>
                          </Checkbox>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      <AccessChooser
        label="Action"
        mode={mode}
        picked={picked}
        tools={tools}
        onChange={(nextMode, nextPicked) => {
          setMode(nextMode)
          setPicked(nextPicked)
        }}
      />
    </FormDialog>
  )
}
