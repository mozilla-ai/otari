import { useState } from "react"

import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import { AccessChooser } from "@/features/tools/AccessChooser"
import { McpNotice } from "@/features/tools/McpNotice"
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
  initialUserIds = [],
  initialQuery = "",
  tools,
  descriptions,
  isKnown,
  onCheck,
  isChecking,
  checkError = "",
  onDismissCheckError,
  readTools,
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
  /** Users already ticked when the dialog opens, such as one picked from Connected users. */
  initialUserIds?: readonly string[]
  /** What the search starts with, so a preselected user is the one in view. */
  initialQuery?: string
  /** What the server's last check listed, to tick from. */
  tools: readonly string[]
  descriptions?: Readonly<Record<string, string>>
  /** Whether a passing check listed them; typed names only until one has. */
  isKnown: boolean
  /** Runs the server's check, offered while its tools are not known. */
  onCheck?: () => void
  isChecking?: boolean
  /** Why the last check started here could not run; `""` when it did. */
  checkError?: string
  onDismissCheckError?: () => void
  /** The tools a manager labeled read, which "Read tools only" allows. */
  readTools: readonly string[]
  isPending: boolean
  error: unknown
  onSubmit: (draft: GrantDraft) => void
}) {
  const seed = accessDraft(editing?.tools ?? ["*"])
  const [mode, setMode] = useState<AccessMode>(seed.mode)
  const [picked, setPicked] = useState<string[]>(seed.picked)
  const [query, setQuery] = useState(initialQuery)
  const [keyIds, setKeyIds] = useState<readonly string[]>([])
  const [userIds, setUserIds] = useState<readonly string[]>(initialUserIds)

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
          : "Pick keys or users. A user covers every key they own."
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
        keyIds.length > 0 ||
        userIds.length !== initialUserIds.length ||
        userIds.some((id) => !initialUserIds.includes(id)) ||
        mode !== seed.mode ||
        picked.join() !== seed.picked.join()
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
      {checkError ? (
        <McpNotice
          message={checkError}
          onDismiss={() => onDismissCheckError?.()}
        />
      ) : null}
      <AccessChooser
        label="Action"
        mode={mode}
        picked={picked}
        tools={tools}
        descriptions={descriptions}
        isKnown={isKnown}
        onCheck={onCheck}
        isChecking={isChecking}
        readTools={readTools}
        onChange={(nextMode, nextPicked) => {
          setMode(nextMode)
          setPicked(nextPicked)
        }}
      />
    </FormDialog>
  )
}
