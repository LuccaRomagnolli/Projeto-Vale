const state = {
  view: "board",
  filters: null,
  overview: null,
  pendingAction: null,
};

const titles = {
  board: ["Turno operacional", "Prioridade Don't Go · próximas 4h"],
  alerts: ["Telemetria crítica", "Alertas Don't Go para tratativa"],
  processing: ["Apontamentos", "Processamento da frota e ciclos"],
  equipment: ["Ficha do equipamento", "Histórico por Tag"],
  actions: ["Fila de trabalho", "Tratativas registradas pela engenharia"],
  performance: ["Confiabilidade", "Desempenho operacional do modelo"],
};

const $ = (id) => document.getElementById(id);

// `note` e `operator` sao texto livre digitado pela operacao e chegam aqui via
// innerHTML. Sem escape, uma tratativa com `<img onerror=...>` executava para
// todos os demais operadores que abrissem o painel.
function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmtInt(value) {
  return Number(value || 0).toLocaleString("pt-BR");
}
function fmtPct(value) {
  return `${((Number(value) || 0) * 100).toFixed(1).replace(".", ",")}%`;
}
function fmtNum(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits).replace(".", ",");
}
function fmtTime(value) {
  if (!value) return "—";
  return String(value).replace("T", " ").replace("+00:00", " UTC").slice(0, 19);
}
function riskClass(code) {
  if (String(code).includes("alto")) return "risk-high";
  if (String(code).includes("medio")) return "risk-mid";
  return "risk-low";
}
function riskLabel(code) {
  if (String(code).includes("alto")) return "Alto";
  if (String(code).includes("medio")) return "Médio";
  return "Baixo";
}
function riskChip(item) {
  const detail = item.risco_rotulo || item.risco_segmento || "";
  return `<span class="badge ${riskClass(item.risco_segmento)}" title="${esc(detail)}">${riskLabel(item.risco_segmento)}</span>`;
}
function statusChip(status) {
  const value = status || "—";
  return `<span class="badge chip-status status-${value}">${esc(String(value).replace(/_/g, " "))}</span>`;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

function fillSelect(select, values, keepEmpty = true) {
  const current = select.value;
  select.innerHTML = keepEmpty ? '<option value="">Todas</option>' : "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
  if ([...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
}

async function loadFilters() {
  state.filters = await api("/api/filters");
  fillSelect($("filter-date"), state.filters.dates, false);
  if (state.filters.latest_date) $("filter-date").value = state.filters.latest_date;
  fillSelect($("filter-frota"), state.filters.frotas);
  fillSelect($("filter-tag"), state.filters.tags);
}

async function loadOverview() {
  const date = $("filter-date").value;
  state.overview = await api(`/api/overview${date ? `?date=${date}` : ""}`);
  const k = state.overview.kpis;
  $("kpis").innerHTML = [
    ["Prioridade do dia", fmtInt(k.priority_count)],
    ["Risco alto", fmtInt(k.high_risk_count)],
    ["Ciclos no dia", fmtInt(k.cycle_count)],
    ["Alvo 4h", fmtInt(k.positive_count)],
    ["Eventos críticos", fmtInt(k.critical_events)],
    ["Tratativas abertas", fmtInt(k.open_actions)],
  ]
    .map(([label, value]) => `<article class="kpi"><span>${label}</span><strong>${value}</strong></article>`)
    .join("");
  const model = state.overview.model;
  $("model-meta").innerHTML = `
    <div>Modelo: <strong>${model.name || "não promovido"}</strong></div>
    <div>Limiar: ${fmtNum(model.threshold, 4)}</div>
    <div>Janela: ${model.horizon} · Top ${model.top_k}</div>
  `;
}

function setView(view) {
  state.view = view;
  document.querySelectorAll(".nav-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  const [eyebrow, title] = titles[view];
  $("view-eyebrow").textContent = eyebrow;
  $("view-title").textContent = title;
  render();
}

function actionButton(item) {
  const payload = encodeURIComponent(JSON.stringify(item));
  return `<button type="button" data-action='${payload}'>Tratar</button>`;
}

function renderTable(headers, rows, emptyMessage) {
  if (!rows.length) return `<div class="empty">${emptyMessage}</div>`;
  return `
    <div class="card table-card">
      <div class="table-wrap">
        <table>
          <thead><tr>${headers
            .map((h) => (typeof h === "object" ? `<th class="${h.cls || ""}">${h.label}</th>` : `<th>${h}</th>`))
            .join("")}</tr></thead>
          <tbody>${rows.join("")}</tbody>
        </table>
      </div>
    </div>`;
}

async function renderBoard() {
  const date = $("filter-date").value;
  const frota = $("filter-frota").value;
  const params = new URLSearchParams({ date, frota });
  const data = await api(`/api/priority?${params}`);
  const bars = Object.entries(data.risk_counts || {})
    .map(([label, count]) => {
      const max = Math.max(...Object.values(data.risk_counts), 1);
      return `<div class="bar-row"><span>${label}</span><div class="bar"><i style="width:${(count / max) * 100}%"></i></div><b>${count}</b></div>`;
    })
    .join("");
  const rows = data.items.map((item) => `
    <tr>
      <td>${item.rank}</td>
      <td><a href="#" data-tag="${esc(item.Tag)}">${esc(item.Tag)}</a></td>
      <td class="nowrap">${esc(item.Frota) || "—"}</td>
      <td class="nowrap">${esc(item.turno) || "—"}</td>
      <td class="num">${fmtNum(item.score, 3)}</td>
      <td>${riskChip(item)}</td>
      <td><span class="clamp">${esc(item.motivo_principal) || "—"}</span></td>
      <td><span class="clamp">${esc(item.acao_recomendada) || "—"}</span></td>
      <td>${statusChip(item.status)}</td>
      <td>${actionButton({ item_type: "priority", tag: item.Tag, date: item.data, status: item.status, note: item.note, operator: item.operator })}</td>
    </tr>`);
  $("content").innerHTML = `
    <div class="grid-2 board-layout">
      ${renderTable(
        ["#", "Tag", "Frota", "Turno", { label: "Score", cls: "num" }, "Risco", "Motivo", "Ação recomendada", "Status", ""],
        rows,
        "Sem ranking para a data. Execute a inferência do pipeline."
      )}
      <aside class="card panel-accent">
        <h3>Composição de risco</h3>
        <div class="bars">${bars || '<p class="muted">Sem itens no turno.</p>'}</div>
        <p class="muted">A lista Top 15 aponta os equipamentos a inspecionar primeiro para reduzir Don't Go nas próximas 4 horas.</p>
      </aside>
    </div>`;
}

async function renderAlerts() {
  const date = $("filter-date").value;
  const tag = $("filter-tag").value;
  const params = new URLSearchParams({ date, tag, page: "1", page_size: "80" });
  const data = await api(`/api/alerts?${params}`);
  const rows = data.items.map((item) => `
    <tr>
      <td><a href="#" data-tag="${esc(item.Tag)}">${esc(item.Tag)}</a></td>
      <td>${fmtTime(item.EVENT_TIME)}</td>
      <td>${item.data || "—"}</td>
      <td>${statusChip(item.status)}</td>
      <td><span class="clamp">${esc(item.note) || "—"}</span></td>
      <td>${actionButton({ item_type: "event", tag: item.Tag, event_time: item.EVENT_TIME, status: item.status, note: item.note })}</td>
    </tr>`);
  $("content").innerHTML = `
    <p class="muted">${fmtInt(data.total)} eventos críticos no recorte.</p>
    ${renderTable(["Tag", "Horário", "Data", "Status", "Nota", ""], rows, "Nenhum alerta crítico no recorte.")}`;
}

async function renderProcessing() {
  const date = $("filter-date").value;
  const tag = $("filter-tag").value;
  const frota = $("filter-frota").value;
  const summary = await api(`/api/processing?date=${date}`);
  const cycles = await api(`/api/cycles?${new URLSearchParams({ date, tag, frota, page: "1", page_size: "80" })}`);
  const maxFleet = Math.max(...summary.by_fleet.map((row) => row.ciclos || 0), 1);
  const fleetBars = summary.by_fleet
    .map((row) => `<div class="bar-row"><span>${row.Frota || "—"}</span><div class="bar"><i style="width:${(row.ciclos / maxFleet) * 100}%"></i></div><b>${row.ciclos}</b></div>`)
    .join("");
  const classRows = summary.by_class.map((row) => `
    <tr><td>${row.Classe || "Sem classe"}</td><td>${fmtInt(row.ciclos)}</td><td>${fmtInt(row.positivos)}</td></tr>`);
  const cycleRows = cycles.items.map((item) => `
    <tr>
      <td>${item.Id}</td>
      <td><a href="#" data-tag="${esc(item.Tag)}">${esc(item.Tag)}</a></td>
      <td>${esc(item.Frota) || "—"}</td>
      <td>${esc(item.Classe) || "—"}</td>
      <td>${fmtTime(item.Inicio)}</td>
      <td>${fmtTime(item.Fim)}</td>
      <td>${fmtNum(item.duracao_ciclo_min, 1)} min</td>
      <td>${item.target_4h ? "Sim" : "Não"}</td>
      <td>${fmtNum(item.tte_horas, 2)}</td>
    </tr>`);
  $("content").innerHTML = `
    <div class="grid-3">
      <article class="card"><h3>Volume por frota</h3><div class="bars">${fleetBars || '<p class="muted">Sem ciclos.</p>'}</div></article>
      <article class="card"><h3>Classes de atividade</h3><table><thead><tr><th>Classe</th><th>Ciclos</th><th>Alvo 4h</th></tr></thead><tbody>${classRows.join("") || '<tr><td colspan="3">Sem dados</td></tr>'}</tbody></table></article>
      <article class="card">
        <h3>Duração do ciclo</h3>
        <p>Mediana: <strong>${fmtNum(summary.duration.median, 1)} min</strong></p>
        <p>P95: <strong>${fmtNum(summary.duration.p95, 1)} min</strong></p>
        <p class="muted">Use duração atípica como evidência de gargalo ou degradação operacional.</p>
      </article>
    </div>
    <p class="muted" style="margin-top:16px">${fmtInt(cycles.total)} ciclos no recorte.</p>
    ${renderTable(["Id", "Tag", "Frota", "Classe", "Início", "Fim", "Duração", "Alvo 4h", "TTE h"], cycleRows, "Sem ciclos processados no recorte.")}`;
}

async function renderEquipment() {
  const tag = $("filter-tag").value || (state.filters.tags[0] || "");
  if (!$("filter-tag").value && tag) $("filter-tag").value = tag;
  if (!tag) {
    $("content").innerHTML = '<div class="empty">Nenhuma Tag disponível.</div>';
    return;
  }
  const data = await api(`/api/equipment/${encodeURIComponent(tag)}`);
  const hotspot = data.hotspot || {};
  const history = data.priority_history.map((item) => `
    <tr><td>${item.data}</td><td class="num">${item.rank}</td><td class="num">${fmtNum(item.score, 3)}</td><td>${riskChip(item)}</td><td><span class="clamp">${esc(item.motivo_principal) || "—"}</span></td></tr>`);
  const alerts = data.recent_alerts.map((item) => `<tr><td>${fmtTime(item.EVENT_TIME)}</td></tr>`);
  const cycles = data.recent_cycles.map((item) => `
    <tr><td>${fmtTime(item.Fim)}</td><td>${esc(item.Classe) || "—"}</td><td>${fmtNum(item.duracao_ciclo_min, 1)}</td><td>${item.target_4h ? "Sim" : "Não"}</td></tr>`);
  $("content").innerHTML = `
    <div class="equipment-search">
      <p class="muted">${data.context.Tipo || "Equipamento"} · ${data.context.Frota || "Frota não informada"} · ciclos ${fmtInt(data.totals.cycles)} · alertas ${fmtInt(data.totals.alerts)} · positivos ${fmtInt(data.totals.positives)}</p>
    </div>
    <div class="grid-3">
      <article class="card">
        <h3>Hotspot no teste</h3>
        <p>Dias positivos: <strong>${fmtInt(hotspot.positive_days)}</strong></p>
        <p>Selecionada no TopK: <strong>${fmtInt(hotspot.selected_days)}</strong></p>
        <p>Precisão na Tag: <strong>${fmtPct(hotspot.selected_precision)}</strong></p>
      </article>
      <article class="card"><h3>Últimas prioridades</h3><table><thead><tr><th>Data</th><th>#</th><th>Score</th><th>Risco</th><th>Motivo</th></tr></thead><tbody>${history.join("") || '<tr><td colspan="5">Sem ranking</td></tr>'}</tbody></table></article>
      <article class="card"><h3>Alertas recentes</h3><table><thead><tr><th>Horário</th></tr></thead><tbody>${alerts.join("") || '<tr><td>Sem eventos</td></tr>'}</tbody></table></article>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>Ciclos recentes</h3>
      <table><thead><tr><th>Fim</th><th>Classe</th><th>Duração min</th><th>Alvo 4h</th></tr></thead><tbody>${cycles.join("") || '<tr><td colspan="4">Sem ciclos</td></tr>'}</tbody></table>
    </div>`;
}

async function renderActions() {
  const data = await api("/api/actions");
  const rows = data.items.map((item) => `
    <tr>
      <td>${item.item_type}</td>
      <td>${item.tag || "—"}</td>
      <td>${item.date || fmtTime(item.event_time)}</td>
      <td>${statusChip(item.status)}</td>
      <td>${esc(item.operator) || "—"}</td>
      <td><span class="clamp">${esc(item.note) || "—"}</span></td>
      <td>${fmtTime(item.updated_at)}</td>
    </tr>`);
  $("content").innerHTML = renderTable(
    ["Tipo", "Tag", "Referência", "Status", "Responsável", "Nota", "Atualizado"],
    rows,
    "Nenhuma tratativa registrada ainda."
  );
}

async function renderPerformance() {
  const data = await api("/api/performance");
  const topk = (data.test_daily_topk || []).map((row) => `
    <tr>
      <td>${row.top_k_tags_per_day}</td>
      <td>${fmtPct(row.precision_at_k)}</td>
      <td>${fmtPct(row.recall_at_k)}</td>
      <td>${fmtNum(row.lift_vs_random, 2)}</td>
      <td>${fmtInt(row.positives_captured)} / ${fmtInt(row.total_positives)}</td>
    </tr>`);
  const hotspots = (data.hotspots || []).map((row) => `
    <tr>
      <td><a href="#" data-tag="${esc(row.Tag)}">${esc(row.Tag)}</a></td>
      <td>${fmtInt(row.positive_days)}</td>
      <td>${fmtInt(row.selected_days)}</td>
      <td>${fmtPct(row.selected_precision)}</td>
      <td>${fmtNum(row.avg_score, 3)}</td>
    </tr>`);
  $("content").innerHTML = `
    <div class="grid-2">
      <article class="card">
        <h3>${data.model.name || "Modelo"}</h3>
        <p>Limiar operacional: <strong>${fmtNum(data.model.threshold, 4)}</strong></p>
        <p class="muted">${data.model.selection_rule || ""}</p>
        <table>
          <thead><tr><th>Top K</th><th>Precision</th><th>Recall</th><th>Lift</th><th>Captura</th></tr></thead>
          <tbody>${topk.join("")}</tbody>
        </table>
      </article>
      <article class="card">
        <h3>Hotspots por Tag</h3>
        <table>
          <thead><tr><th>Tag</th><th>Dias +</th><th>Selecionada</th><th>Precisão</th><th>Score médio</th></tr></thead>
          <tbody>${hotspots.join("")}</tbody>
        </table>
      </article>
    </div>`;
}

async function render() {
  $("content").innerHTML = '<div class="empty">Carregando…</div>';
  const views = {
    board: renderBoard,
    alerts: renderAlerts,
    processing: renderProcessing,
    equipment: renderEquipment,
    actions: renderActions,
    performance: renderPerformance,
  };
  try {
    await loadOverview();
    await views[state.view]();
  } catch (error) {
    $("content").innerHTML = `<div class="empty">Falha ao carregar a interface: ${error.message}</div>`;
  }
}

function openAction(item) {
  state.pendingAction = item;
  $("dialog-item").textContent = `${item.item_type} · ${item.tag || item.cycle_id || ""}`;
  $("action-status").value = item.status || "em_inspecao";
  $("action-operator").value = item.operator || "";
  $("action-note").value = item.note || "";
  $("action-dialog").showModal();
}

document.querySelector(".nav").addEventListener("click", (event) => {
  const button = event.target.closest(".nav-btn");
  if (button) setView(button.dataset.view);
});

$("content").addEventListener("click", (event) => {
  const tagLink = event.target.closest("[data-tag]");
  if (tagLink) {
    event.preventDefault();
    $("filter-tag").value = tagLink.dataset.tag;
    setView("equipment");
    return;
  }
  const actionBtn = event.target.closest("[data-action]");
  if (actionBtn) openAction(JSON.parse(actionBtn.dataset.action));
});

$("filter-date").addEventListener("change", render);
$("filter-frota").addEventListener("change", render);
$("filter-tag").addEventListener("change", () => {
  if (state.view === "board") render();
  else render();
});
$("reload-btn").addEventListener("click", async () => {
  await api("/api/reload", { method: "POST" });
  await loadFilters();
  render();
});

$("action-save").addEventListener("click", async (event) => {
  event.preventDefault();
  const item = state.pendingAction;
  if (!item) return;
  await api("/api/actions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...item,
      status: $("action-status").value,
      operator: $("action-operator").value,
      note: $("action-note").value,
    }),
  });
  $("action-dialog").close();
  render();
});

(async function init() {
  await loadFilters();
  setView("board");
})();
