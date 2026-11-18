import type { IconType } from "react-icons"
import {
  FiEdit2,
  FiLayers,
  FiPause,
  FiPlay,
  FiRefreshCw,
  FiTrash2,
  FiUserCheck,
  FiUsers,
} from "react-icons/fi"
import type { ApiKey } from "@/client"
import { RowMenu } from "@/design-system/actions/RowMenu"
import {
  type KeyMenuId,
  type KeyMenuItem,
  keyLabel,
} from "@/features/keys/keyList"
import { keyFingerprint } from "@/features/keys/secretCaption"
import { formatDate } from "@/shared/helpers/format"

const ICONS: Record<KeyMenuId, IconType> = {
  edit: FiEdit2,
  disable: FiPause,
  enable: FiPlay,
  "end-users": FiUsers,
  "workload-identity": FiUserCheck,
  regenerate: FiRefreshCw,
  sessions: FiLayers,
  delete: FiTrash2,
}

/**
 * A key row's menu: which key it is (its whole fingerprint, where it was
 * made, and what it may reach), then what the caller may do to it. A caller
 * who may change nothing is told why at its foot.
 */
export function KeyRowMenu({
  apiKey,
  items,
  isReadOnly,
  teamName,
  organizationName,
  budget,
  onAction,
}: {
  apiKey: ApiKey
  items: readonly KeyMenuItem[]
  isReadOnly: boolean
  teamName: string | undefined
  organizationName: string | undefined
  /** What the key has spent against its own cap, for a caller who may cap keys. */
  budget?: string
  onAction: (id: KeyMenuId) => void
}) {
  const name = keyLabel(apiKey)
  const origin = apiKey.is_organization_service_key
    ? `Service key of ${organizationName ?? "the organization"}`
    : `Created ${formatDate(apiKey.created_at)}${teamName ? ` in ${teamName}` : ""}`
  return (
    <RowMenu
      label={`Actions for ${name}`}
      head={
        <div className="flex flex-col gap-1 border-b border-border p-3">
          <p className="break-all text-mono-caption text-muted">
            {keyFingerprint(apiKey) ?? name}
          </p>
          <p className="text-caption">{origin}</p>
          <dl className="flex flex-col gap-1 pt-1 text-caption">
            <div>
              <dt className="inline">Models: </dt>
              <dd className="inline break-words">
                {apiKey.allowed_models === null
                  ? "All models"
                  : apiKey.allowed_models.join(", ") || "No models"}
              </dd>
            </div>
            {budget ? (
              <div>
                <dt className="inline">Budget: </dt>
                <dd className="inline">{budget}</dd>
              </div>
            ) : null}
            {apiKey.exclude_from_budget ? (
              <div>
                <dt className="inline">Budget: </dt>
                <dd className="inline">Exempt</dd>
              </div>
            ) : null}
            {apiKey.reject_user_mismatch !== null ? (
              <div>
                <dt className="inline">User matching: </dt>
                <dd className="inline">
                  {apiKey.reject_user_mismatch ? "Strict" : "Lenient"}
                </dd>
              </div>
            ) : null}
          </dl>
        </div>
      }
      foot={
        isReadOnly ? (
          <p className="border-t border-border-subtle px-3 py-2.5 text-caption">
            A viewer cannot change or delete a key, including their own.
          </p>
        ) : undefined
      }
      actions={items.map((item) => ({
        ...item,
        icon: ICONS[item.id],
        run: () => onAction(item.id),
      }))}
    />
  )
}
