// The second gateway this suite boots, in hybrid mode (e2e/serve-hybrid.sh),
// shared by playwright.config.ts and the screenshot registry that captures its
// landing page. The port is e2e/otari.hybrid.yml's, and the host is 127.0.0.1
// rather than localhost so the page's own `window.location.origin` matches this
// string, which parity.hybrid.spec.ts asserts against.
export const HYBRID_BASE_URL = "http://127.0.0.1:8010"
