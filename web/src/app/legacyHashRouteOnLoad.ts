/**
 * Runs `installLegacyHashRoute` as this module is evaluated. `main.tsx` imports
 * it first, so it runs before the router module reads the address.
 */
import { installLegacyHashRoute } from "./legacyHashRoute"

installLegacyHashRoute(window, import.meta.env.BASE_URL)
