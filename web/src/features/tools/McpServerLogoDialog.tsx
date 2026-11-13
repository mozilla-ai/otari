import { useEffect, useState } from "react"
import { FileTrigger } from "react-aria-components"
import { FiUpload } from "react-icons/fi"

import type { WorkspaceMcpServerLogo } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { RadioGroup, type RadioOption } from "@/design-system/forms/RadioGroup"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import {
  type McpLogoTheme,
  type McpLogoTone,
  McpServerLogo,
} from "@/features/tools/McpServerLogo"
import {
  logoTone,
  type McpLogoTile,
  type PreparedLogo,
  prepareLogo,
} from "@/features/tools/mcpLogoImage"
import {
  useRemoveWorkspaceMcpServerLogo,
  useSetWorkspaceMcpServerLogo,
} from "@/shared/api/tools"

const TILES: RadioOption[] = [
  { value: "auto", label: "Auto" },
  { value: "always", label: "Always" },
  { value: "never", label: "Never" },
]

/** The server as a row shows it on one theme, whatever theme the admin is using. */
function ThemePreview({
  theme,
  name,
  logoUrl,
  tone,
}: {
  theme: McpLogoTheme
  name: string
  logoUrl: string | null | undefined
  tone: McpLogoTone | undefined
}) {
  return (
    <div
      data-theme={theme}
      className="flex min-w-0 flex-1 flex-col gap-2 rounded-md border border-border bg-surface p-3 text-foreground"
    >
      <span className="text-caption">
        {theme === "light" ? "Light" : "Dark"}
      </span>
      <div className="flex min-w-0 items-center gap-3">
        <McpServerLogo
          name={name}
          logoUrl={logoUrl}
          tone={tone}
          theme={theme}
          size="lg"
        />
        <span className="truncate font-medium">{name}</span>
      </div>
    </div>
  )
}

/**
 * Upload a logo for one of the workspace's servers, choose whether it sits on a
 * tile, and see it on both themes before saving. The upload is cropped to a
 * square and drawn at 256 px here; the gateway checks it again and keeps it
 * apart from the logo it fetches, so nothing replaces it but an admin.
 */
export function McpServerLogoDialog({
  isOpen,
  onOpenChange,
  workspaceId,
  serverId,
  name,
  logo,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  workspaceId: string
  serverId: string
  name: string
  /** What the server shows now; absent while it loads. */
  logo: WorkspaceMcpServerLogo | undefined
}) {
  const save = useSetWorkspaceMcpServerLogo()
  const remove = useRemoveWorkspaceMcpServerLogo()
  const current: McpLogoTile = logo?.logo_tile ?? "auto"
  const [tile, setTile] = useState<McpLogoTile>(current)
  const [upload, setUpload] = useState<PreparedLogo>()
  const [uploadError, setUploadError] = useState("")

  useEffect(
    () => () => {
      if (upload) URL.revokeObjectURL(upload.previewUrl)
    },
    [upload],
  )

  const close = () => {
    setUpload(undefined)
    setUploadError("")
    setTile(current)
    save.reset()
    remove.reset()
    onOpenChange(false)
  }

  const pick = async (file: File | undefined) => {
    if (!file) return
    try {
      setUpload(await prepareLogo(file))
      setUploadError("")
    } catch (error) {
      setUploadError(errorMessage(error))
    }
  }

  // What the rows will show: the new upload, else what they show today.
  const previewUrl = upload?.previewUrl ?? logo?.logo_url
  let tone: McpLogoTone | undefined
  if (upload) tone = logoTone(tile, upload.luminance)
  else if (tile === current) tone = logo?.logo_tone ?? undefined
  else if (tile !== "auto") tone = logoTone(tile, undefined)
  else tone = undefined

  const isBusy = save.isPending || remove.isPending
  const hasChanges = upload !== undefined || tile !== current
  const ids = { workspaceId, serverId }

  return (
    <Dialog
      isOpen={isOpen}
      onOpenChange={(next) => (next ? onOpenChange(true) : close())}
      isDismissable={!isBusy}
      title={`Logo for ${name}`}
      footerStart={
        logo?.has_logo_override ? (
          <Button
            variant="danger"
            size="sm"
            isPending={remove.isPending}
            isDisabled={save.isPending}
            onPress={() => remove.mutate(ids, { onSuccess: close })}
          >
            Remove uploaded logo
          </Button>
        ) : undefined
      }
      actions={
        <>
          <Button isDisabled={isBusy} onPress={close}>
            Cancel
          </Button>
          <Button
            variant="primary"
            isDisabled={!hasChanges || remove.isPending}
            isPending={save.isPending}
            onPress={() =>
              save.mutate(
                { ...ids, tile, png: upload?.png },
                { onSuccess: close },
              )
            }
          >
            Save
          </Button>
        </>
      }
    >
      <DialogSection>
        <div className="flex flex-col gap-4">
          <ErrorBanner error={save.error ?? remove.error} />
          <div className="flex flex-col gap-3 sm:flex-row">
            <ThemePreview
              theme="light"
              name={name}
              logoUrl={previewUrl}
              tone={tone}
            />
            <ThemePreview
              theme="dark"
              name={name}
              logoUrl={previewUrl}
              tone={tone}
            />
          </div>
          <div className="flex flex-col items-start gap-1">
            <FileTrigger
              acceptedFileTypes={["image/png", "image/svg+xml"]}
              onSelect={(files) => void pick(files?.[0])}
            >
              <Button size="sm" isDisabled={isBusy}>
                <FiUpload aria-hidden className="size-4" />
                {upload ? "Choose another file" : "Choose a PNG or SVG"}
              </Button>
            </FileTrigger>
            <span className="text-caption">
              Cropped to a square and resized to 256 px. Up to 512 KB.
            </span>
            {uploadError ? (
              <span role="alert" className="text-caption text-danger">
                {uploadError}
              </span>
            ) : null}
          </div>
          <RadioGroup
            label="Tile behind the logo"
            orientation="horizontal"
            value={tile}
            options={TILES}
            isDisabled={isBusy}
            description="Auto adds one only on the theme where the logo would fade out."
            onChange={(next) => setTile(next as McpLogoTile)}
          />
        </div>
      </DialogSection>
    </Dialog>
  )
}

/**
 * A server's logo as a control: for an admin, it opens the logo dialog. Plain
 * at rest; a ring on hover or keyboard focus says it can be pressed.
 */
export function McpServerLogoButton({
  workspaceId,
  serverId,
  name,
  logo,
  size = "lg",
}: {
  workspaceId: string
  serverId: string
  name: string
  logo: WorkspaceMcpServerLogo | undefined
  size?: "sm" | "md" | "lg"
}) {
  const [isOpen, setOpen] = useState(false)
  return (
    <>
      <Tooltip content="Change logo">
        {(trigger) => (
          <button
            {...trigger}
            type="button"
            aria-label={`Change the logo of ${name}`}
            className="flex shrink-0 cursor-pointer rounded-md outline-none hover:otari-focus-ring focus-visible:otari-focus-ring"
            onClick={(event) => {
              trigger.onClick?.(event)
              setOpen(true)
            }}
          >
            <McpServerLogo
              name={name}
              logoUrl={logo?.logo_url}
              tone={logo?.logo_tone}
              size={size}
            />
          </button>
        )}
      </Tooltip>
      <McpServerLogoDialog
        isOpen={isOpen}
        onOpenChange={setOpen}
        workspaceId={workspaceId}
        serverId={serverId}
        name={name}
        logo={logo}
      />
    </>
  )
}
