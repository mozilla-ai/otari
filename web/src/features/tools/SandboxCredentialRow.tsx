import { useState } from "react"
import type { SandboxCredentialPublic } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { INPUT_CLASS } from "@/design-system/forms/inputClass"
import { SettingRow } from "@/design-system/layout/SettingRow"
import {
  useClearSandboxCredential,
  useSetSandboxCredential,
} from "@/shared/api/tools"
import { commitOnEnter, useAutosave } from "@/shared/hooks/useAutosave"

function sourceHelp(credential: SandboxCredentialPublic): string {
  if (credential.source === "stored") {
    return `Stored here, ending ${credential.last4}. It wins over ${credential.env_var} and reaches every replica within 30 seconds.`
  }
  if (credential.source === "environment") {
    return `From ${credential.env_var}, ending ${credential.last4}. A key stored here wins over it, with no restart.`
  }
  return `No key: code execution fails until one is stored here or set in ${credential.env_var}.`
}

/** The hosted sandbox provider's key: written here, never read back. */
export function SandboxCredentialRow({
  credential,
}: {
  credential: SandboxCredentialPublic
}) {
  const store = useSetSandboxCredential()
  const clear = useClearSandboxCredential()
  const save = useAutosave()
  const [apiKey, setApiKey] = useState("")
  const isBusy = save.isSaving

  return (
    <SettingRow
      label="API key"
      controlId="sandbox-api-key"
      help={sourceHelp(credential)}
      error={save.error}
      errorId="sandbox-api-key-error"
      control={
        <div className="flex w-full items-center gap-2">
          <input
            id="sandbox-api-key"
            type="password"
            autoComplete="new-password"
            aria-invalid={save.error ? true : undefined}
            aria-describedby={save.error ? "sandbox-api-key-error" : undefined}
            value={apiKey}
            disabled={isBusy}
            placeholder={
              credential.source === "stored"
                ? `replace key ····${credential.last4}`
                : "add key"
            }
            onChange={(event) => setApiKey(event.target.value)}
            onKeyDown={commitOnEnter}
            // Blank means "leave the stored key alone", so leaving an empty
            // field writes nothing.
            onBlur={() => {
              if (apiKey.trim() === "") return
              void save.run(async () => {
                await store.mutateAsync({ api_key: apiKey })
                setApiKey("")
              })
            }}
            className={`otari-machine-field min-w-0 flex-1 ${INPUT_CLASS}`}
          />
          {credential.source === "stored" ? (
            <Button
              variant="ghost"
              aria-label="Clear the stored sandbox key"
              // Disabled while a key is typed: pressing Clear blurs the field,
              // which saves that key, and the two writes would race.
              isDisabled={isBusy || apiKey !== ""}
              onPress={() => void save.run(() => clear.mutateAsync(undefined))}
            >
              Clear
            </Button>
          ) : null}
        </div>
      }
    />
  )
}
