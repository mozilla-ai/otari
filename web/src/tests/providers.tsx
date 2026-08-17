/**
 * The app's real provider tree, re-exported for tests that need it.
 *
 * `src/tests/` sits outside the layer boundary on purpose (see biome.jsonc): a
 * harness has to reach the composition root to mount what the app mounts, and a
 * feature or a shared module never does. Going through here is what lets a
 * feature's test render against the real providers without importing `@/app`
 * itself. Most page tests need neither and mock `@/shared/api/hooks` outright.
 */
export { Provider as AppProviders } from "@/app/provider";
