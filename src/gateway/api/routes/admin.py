"""Self-hosted operator dashboard for standalone gateway mode."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["admin"])

_ADMIN_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Otari Gateway Admin</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f5f6f8;
        --panel: #ffffff;
        --panel-muted: #eef1f5;
        --line: #d6dce5;
        --text: #111827;
        --muted: #5b6472;
        --accent: #2563eb;
        --ok: #147d64;
        --warn: #b45309;
        --bad: #b91c1c;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 14px;
      }
      header {
        align-items: center;
        background: #101827;
        color: #ffffff;
        display: flex;
        gap: 16px;
        justify-content: space-between;
        padding: 14px 20px;
      }
      h1 {
        font-size: 18px;
        font-weight: 650;
        margin: 0;
      }
      h2 {
        font-size: 14px;
        margin: 0 0 10px;
      }
      main {
        display: grid;
        gap: 14px;
        grid-template-columns: 280px minmax(0, 1fr);
        padding: 14px;
      }
      aside, section, dialog {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
      }
      aside {
        align-self: start;
        padding: 14px;
      }
      section {
        margin-bottom: 14px;
        overflow: hidden;
      }
      .section-head {
        align-items: center;
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        padding: 12px 14px;
      }
      .content { padding: 12px 14px; }
      label {
        color: var(--muted);
        display: block;
        font-size: 12px;
        margin: 0 0 6px;
      }
      input, textarea {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 6px;
        color: var(--text);
        font: inherit;
        min-height: 36px;
        padding: 8px 10px;
        width: 100%;
      }
      textarea {
        min-height: 92px;
        resize: vertical;
      }
      button {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 6px;
        color: var(--text);
        cursor: pointer;
        font: inherit;
        min-height: 34px;
        padding: 7px 10px;
      }
      button.primary {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
      }
      button.warn {
        border-color: #f0b85b;
        color: var(--warn);
      }
      button:disabled {
        cursor: not-allowed;
        opacity: 0.55;
      }
      .stack { display: grid; gap: 10px; }
      .row {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .tabs {
        background: var(--panel-muted);
        border: 1px solid var(--line);
        border-radius: 8px;
        display: grid;
        gap: 4px;
        padding: 4px;
      }
      .tab {
        border: 0;
        border-radius: 6px;
        text-align: left;
      }
      .tab.active {
        background: #ffffff;
        color: var(--accent);
        font-weight: 650;
      }
      .metric-grid {
        display: grid;
        gap: 10px;
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
      .metric {
        background: var(--panel-muted);
        border-radius: 8px;
        padding: 10px;
      }
      .metric strong {
        display: block;
        font-size: 20px;
        margin-top: 4px;
      }
      table {
        border-collapse: collapse;
        width: 100%;
      }
      th, td {
        border-bottom: 1px solid var(--line);
        padding: 9px 10px;
        text-align: left;
        vertical-align: top;
      }
      th {
        color: var(--muted);
        font-size: 12px;
        font-weight: 650;
      }
      tbody tr:hover { background: #f8fafc; }
      .pill {
        border: 1px solid var(--line);
        border-radius: 999px;
        display: inline-block;
        font-size: 12px;
        padding: 2px 8px;
      }
      .ok { color: var(--ok); }
      .bad { color: var(--bad); }
      .muted { color: var(--muted); }
      .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
      }
      .view { display: none; }
      .view.active { display: block; }
      .status {
        color: #d8dee9;
        font-size: 13px;
      }
      dialog {
        max-width: 780px;
        padding: 0;
        width: calc(100vw - 32px);
      }
      dialog::backdrop { background: rgb(15 23 42 / 0.45); }
      pre {
        background: #101827;
        border-radius: 8px;
        color: #f8fafc;
        margin: 0;
        max-height: 420px;
        overflow: auto;
        padding: 12px;
        white-space: pre-wrap;
      }
      @media (max-width: 860px) {
        main { grid-template-columns: 1fr; }
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>Otari Gateway Admin</h1>
      <div class="status" id="status">Disconnected</div>
    </header>
    <main>
      <aside class="stack">
        <div>
          <label for="master-key">Master key</label>
          <input id="master-key" type="password" autocomplete="off" />
        </div>
        <button class="primary" id="refresh">Refresh</button>
        <div class="tabs" id="tabs">
          <button class="tab active" data-view="overview">Overview</button>
          <button class="tab" data-view="policies">Policies</button>
          <button class="tab" data-view="traces">Traces</button>
          <button class="tab" data-view="usage">Usage</button>
          <button class="tab" data-view="alerts">Alerts</button>
        </div>
      </aside>
      <div>
        <section class="view active" id="view-overview">
          <div class="section-head"><h2>Overview</h2><span class="muted" id="loaded-at"></span></div>
          <div class="content metric-grid" id="overview-metrics"></div>
        </section>
        <section class="view" id="view-policies">
          <div class="section-head"><h2>Routing policies</h2><button id="reload-policies">Reload</button></div>
          <div class="content" id="policies-table"></div>
        </section>
        <section class="view" id="view-traces">
          <div class="section-head"><h2>Route traces</h2><button id="reload-traces">Reload</button></div>
          <div class="content" id="traces-table"></div>
        </section>
        <section class="view" id="view-usage">
          <div class="section-head"><h2>Usage summary</h2><button id="reload-usage">Reload</button></div>
          <div class="content" id="usage-summary"></div>
        </section>
        <section class="view" id="view-alerts">
          <div class="section-head"><h2>Budget alerts</h2><button id="reload-alerts">Reload</button></div>
          <div class="content" id="alerts-table"></div>
        </section>
      </div>
    </main>
    <dialog id="modal">
      <div class="section-head">
        <h2 id="modal-title">Details</h2>
        <button id="modal-close">Close</button>
      </div>
      <div class="content stack" id="modal-body"></div>
    </dialog>
    <script>
      const state = { key: "", policies: [], projects: [], traces: [], alerts: [], usage: null, traceSummary: null };
      const $ = (id) => document.getElementById(id);
      const money = (value) => Number(value || 0).toLocaleString(undefined, {
        maximumFractionDigits: 6,
        style: "currency",
        currency: "USD"
      });
      const number = (value) => Number(value || 0).toLocaleString();
      const safe = (value) => String(value ?? "");
      const jsonBlock = (value) => `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;

      function escapeHtml(value) {
        return safe(value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;");
      }

      function setStatus(message, kind = "") {
        $("status").textContent = message;
        $("status").className = `status ${kind}`;
      }

      function authHeaders(extra = {}) {
        return { ...extra, "Otari-Key": `Bearer ${state.key}` };
      }

      async function api(path, options = {}) {
        if (!state.key) throw new Error("Master key is required");
        const response = await fetch(path, {
          ...options,
          headers: authHeaders(options.headers || {})
        });
        const text = await response.text();
        const payload = text ? JSON.parse(text) : null;
        if (!response.ok) {
          const detail = payload && payload.detail ? payload.detail : response.statusText;
          throw new Error(`${response.status} ${detail}`);
        }
        return payload;
      }

      function renderMetrics() {
        const usage = state.usage || {};
        const traceSummary = state.traceSummary || {};
        const metrics = [
          ["Policies", state.policies.length],
          ["Projects", state.projects.length],
          ["Route traces", traceSummary.total_count || 0],
          ["Usage cost", money(usage.cost || 0)],
          ["Usage tokens", number(usage.total_tokens || 0)],
          ["Successful traces", number(traceSummary.success_count || 0)],
          ["Errored traces", number(traceSummary.error_count || 0)],
          ["Budget alerts", state.alerts.length]
        ];
        $("overview-metrics").innerHTML = metrics.map(([label, value]) => `
          <div class="metric"><span class="muted">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
        `).join("");
        $("loaded-at").textContent = new Date().toLocaleTimeString();
      }

      function table(headers, rows) {
        if (!rows.length) return `<div class="muted">No rows</div>`;
        const head = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
        return `<table><thead><tr>${head}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
      }

      function renderPolicies() {
        const rows = state.policies.map((policy) => `
          <tr>
            <td><div>${escapeHtml(policy.name)}</div><div class="mono muted">${escapeHtml(policy.policy_id)}</div></td>
            <td><span class="pill">${escapeHtml(policy.status)}</span></td>
            <td>${escapeHtml(policy.strategy)}</td>
            <td>${policy.is_default ? '<span class="ok">yes</span>' : '<span class="muted">no</span>'}</td>
            <td>${escapeHtml(policy.revision)}</td>
            <td class="row">
              <button data-revisions="${escapeHtml(policy.policy_id)}">Revisions</button>
              <button data-json="${escapeHtml(policy.policy_id)}">JSON</button>
            </td>
          </tr>
        `);
        $("policies-table").innerHTML = table(["Name", "Status", "Strategy", "Default", "Rev", ""], rows);
      }

      function renderTraces() {
        const rows = state.traces.map((trace) => `
          <tr>
            <td>
              <div class="mono">${escapeHtml(trace.trace_id)}</div>
              <div class="muted">${escapeHtml(trace.endpoint)}</div>
            </td>
            <td>${escapeHtml(trace.status)}</td>
            <td>${escapeHtml(trace.selected_model || "")}</td>
            <td>${escapeHtml(trace.policy_source || "")}</td>
            <td>${money(trace.estimated_cost || 0)}</td>
            <td><button data-trace="${escapeHtml(trace.trace_id)}">Open</button></td>
          </tr>
        `);
        $("traces-table").innerHTML = table(["Trace", "Status", "Selected model", "Source", "Cost", ""], rows);
      }

      function bucketList(title, buckets) {
        const rows = (buckets || []).slice(0, 8).map((bucket) => `
          <tr>
            <td>${escapeHtml(bucket.key)}</td>
            <td>${number(bucket.count)}</td>
            <td>${money(bucket.estimated_cost || bucket.cost || 0)}</td>
          </tr>
        `);
        return `<h2>${escapeHtml(title)}</h2>${table(["Key", "Count", "Cost"], rows)}`;
      }

      function renderUsage() {
        const usage = state.usage || {};
        $("usage-summary").innerHTML = `
          <div class="metric-grid">
            <div class="metric">
              <span class="muted">Requests</span><strong>${number(usage.total_count || 0)}</strong>
            </div>
            <div class="metric">
              <span class="muted">Tokens</span><strong>${number(usage.total_tokens || 0)}</strong>
            </div>
            <div class="metric"><span class="muted">Cost</span><strong>${money(usage.cost || 0)}</strong></div>
            <div class="metric">
              <span class="muted">Errors</span><strong>${number(usage.error_count || 0)}</strong>
            </div>
          </div>
          <div class="stack" style="margin-top: 12px">
            ${bucketList("By project", usage.by_project)}
            ${bucketList("By provider", usage.by_provider)}
            ${bucketList("By tag", usage.by_tag)}
          </div>
        `;
      }

      function renderAlerts() {
        const rows = state.alerts.map((alert) => `
          <tr>
            <td><div class="mono">${escapeHtml(alert.alert_id || alert.id)}</div></td>
            <td>${escapeHtml(alert.scope_type)}:${escapeHtml(alert.scope_id)}</td>
            <td>${escapeHtml(alert.delivery_status || "")}</td>
            <td>${money(alert.spend || 0)}</td>
            <td>${escapeHtml(alert.created_at || "")}</td>
            <td><button data-alert="${escapeHtml(alert.alert_id || alert.id)}">JSON</button></td>
          </tr>
        `);
        $("alerts-table").innerHTML = table(["Alert", "Scope", "Delivery", "Spend", "Created", ""], rows);
      }

      function showModal(title, html) {
        $("modal-title").textContent = title;
        $("modal-body").innerHTML = html;
        $("modal").showModal();
      }

      async function showRevisions(policyId) {
        const revisions = await api(`/v1/routing-policies/${encodeURIComponent(policyId)}/revisions?limit=20`);
        const rows = revisions.map((revision) => `
          <tr>
            <td>${escapeHtml(revision.revision)}</td>
            <td>${escapeHtml(revision.action)}</td>
            <td>${escapeHtml(revision.status)}</td>
            <td>${escapeHtml(revision.change_note || "")}</td>
            <td>
              <button class="warn" data-apply-revision="${escapeHtml(policyId)}:${escapeHtml(revision.revision)}">
                Apply
              </button>
            </td>
          </tr>
        `);
        showModal("Policy revisions", table(["Rev", "Action", "Status", "Note", ""], rows));
      }

      async function applyRevision(policyId, revision) {
        const body = JSON.stringify({ change_note: `Applied revision ${revision} from admin dashboard` });
        await api(`/v1/routing-policies/${encodeURIComponent(policyId)}/revisions/${revision}/apply`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body
        });
        await loadPolicies();
        renderPolicies();
        setStatus(`Applied ${policyId}@${revision}`, "ok");
      }

      async function loadPolicies() {
        state.policies = await api("/v1/routing-policies?limit=200");
      }

      async function loadTraces() {
        state.traces = await api("/v1/route-traces?limit=25");
        state.traceSummary = await api("/v1/route-traces/summary?limit=1000");
      }

      async function loadUsage() {
        state.usage = await api("/v1/usage/summary?limit=1000");
      }

      async function loadAlerts() {
        state.alerts = await api("/v1/budgets/alerts?limit=25");
      }

      async function refreshAll() {
        state.key = $("master-key").value.trim();
        setStatus("Loading...");
        try {
          const [policies, projects] = await Promise.all([
            api("/v1/routing-policies?limit=200"),
            api("/v1/projects?limit=200")
          ]);
          state.policies = policies;
          state.projects = projects;
          await Promise.all([loadTraces(), loadUsage(), loadAlerts()]);
          renderMetrics();
          renderPolicies();
          renderTraces();
          renderUsage();
          renderAlerts();
          setStatus("Connected", "ok");
        } catch (error) {
          setStatus(error.message, "bad");
        }
      }

      $("refresh").addEventListener("click", refreshAll);
      $("reload-policies").addEventListener("click", async () => { await loadPolicies(); renderPolicies(); });
      $("reload-traces").addEventListener("click", async () => {
        await loadTraces();
        renderTraces();
        renderMetrics();
      });
      $("reload-usage").addEventListener("click", async () => { await loadUsage(); renderUsage(); renderMetrics(); });
      $("reload-alerts").addEventListener("click", async () => {
        await loadAlerts();
        renderAlerts();
        renderMetrics();
      });
      $("modal-close").addEventListener("click", () => $("modal").close());

      $("tabs").addEventListener("click", (event) => {
        const button = event.target.closest("[data-view]");
        if (!button) return;
        document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
        document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
        button.classList.add("active");
        $(`view-${button.dataset.view}`).classList.add("active");
      });

      document.body.addEventListener("click", async (event) => {
        const revisions = event.target.closest("[data-revisions]");
        const json = event.target.closest("[data-json]");
        const trace = event.target.closest("[data-trace]");
        const alert = event.target.closest("[data-alert]");
        const apply = event.target.closest("[data-apply-revision]");
        try {
          if (revisions) await showRevisions(revisions.dataset.revisions);
          if (json) {
            const policy = state.policies.find((item) => item.policy_id === json.dataset.json);
            showModal("Policy JSON", jsonBlock(policy));
          }
          if (trace) showModal("Route trace", jsonBlock(await api(`/v1/route-traces/${trace.dataset.trace}`)));
          if (alert) {
            const item = state.alerts.find((entry) => String(entry.alert_id || entry.id) === alert.dataset.alert);
            showModal("Budget alert", jsonBlock(item));
          }
          if (apply) {
            const [policyId, revision] = apply.dataset.applyRevision.split(":");
            await applyRevision(policyId, revision);
          }
        } catch (error) {
          setStatus(error.message, "bad");
        }
      });
    </script>
  </body>
</html>
"""


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard() -> str:
    """Return the standalone gateway operator dashboard."""
    return _ADMIN_DASHBOARD_HTML


@router.get("/admin/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard_slash() -> str:
    """Return the standalone gateway operator dashboard."""
    return _ADMIN_DASHBOARD_HTML
