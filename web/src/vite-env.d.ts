/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Mixpanel project token. Empty or unset means the SDK is never loaded. */
  readonly VITE_MIXPANEL_TOKEN?: string
}
