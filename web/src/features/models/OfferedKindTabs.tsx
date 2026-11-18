import type { IconType } from "react-icons"
import {
  FiBox,
  FiFilm,
  FiGitBranch,
  FiGrid,
  FiHelpCircle,
  FiImage,
  FiLayers,
  FiMic,
  FiShield,
  FiShuffle,
  FiType,
  FiVolume2,
  FiZap,
} from "react-icons/fi"

import { Tab, TabRow } from "@/design-system/navigation/TabRow"

import { type KindFilter, kindTabs, type OfferedKind } from "./offeredKinds"

const KIND_ICONS: Readonly<Record<OfferedKind, IconType>> = {
  chat: FiType,
  image: FiImage,
  video: FiFilm,
  speech: FiVolume2,
  transcription: FiMic,
  decision: FiGitBranch,
  embedding: FiLayers,
  rerank: FiShuffle,
  moderation: FiShield,
  realtime: FiZap,
  other: FiBox,
  unknown: FiHelpCircle,
}

/** One tab per kind of model a provider offers, with All first, each with its count. */
export function OfferedKindTabs({
  counts,
  shown,
  onShow,
}: {
  /** How many models of each kind, as the list endpoint counts them. */
  counts: Readonly<Record<string, number>> | undefined
  shown: KindFilter
  onShow: (kind: KindFilter) => void
}) {
  return (
    <TabRow>
      {kindTabs(counts, shown).map((tab) => {
        const Icon = tab.kind === "" ? FiGrid : KIND_ICONS[tab.kind]
        return (
          <Tab
            key={tab.kind || "all"}
            isActive={tab.kind === shown}
            onPress={() => onShow(tab.kind)}
          >
            <span className="inline-flex items-center gap-1.5">
              <Icon aria-hidden className="size-3.5" />
              {tab.label}{" "}
              <span className="tabular-nums text-subtle">{tab.count}</span>
            </span>
          </Tab>
        )
      })}
    </TabRow>
  )
}
