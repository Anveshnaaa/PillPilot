const state = {
  role: null,
  analysisId: null,
  data: null,
};
const GEMINI_KEY = window.PILLPILOT_GEMINI_API_KEY || "";

const landingEl = document.getElementById("landing");
const dashboardEl = document.getElementById("dashboard");
const roleHeadingEl = document.getElementById("roleHeading");
const errorMsgEl = document.getElementById("errorMsg");

const distViewEl = document.getElementById("distributorView");
const ownerViewEl = document.getElementById("ownerView");

const ownerStoreSelect = document.getElementById("ownerStoreSelect");
const ownerMedicineSelect = document.getElementById("ownerMedicineSelect");
const distStoreFilter = document.getElementById("distStoreFilter");
const distMedicineSelect = document.getElementById("distMedicineSelect");

function showError(msg) {
  errorMsgEl.textContent = msg;
  errorMsgEl.classList.remove("hidden");
}

function clearError() {
  errorMsgEl.textContent = "";
  errorMsgEl.classList.add("hidden");
}

function switchRole(role) {
  state.role = role;
  landingEl.classList.add("hidden");
  dashboardEl.classList.remove("hidden");
  roleHeadingEl.textContent = `${role} Dashboard`;
  distViewEl.classList.toggle("hidden", role !== "Distributor");
  ownerViewEl.classList.toggle("hidden", role !== "Store Owner");
  renderAssistantWidget();
}

function goBack() {
  state.role = null;
  state.analysisId = null;
  state.data = null;
  dashboardEl.classList.add("hidden");
  landingEl.classList.remove("hidden");
  clearError();
  renderAssistantWidget();
}

function renderKpis(kpis) {
  const target = document.getElementById("kpis");
  target.innerHTML = "";
  const rows = [
    ["Stores Monitored", kpis.stores_monitored],
    ["SKUs In Latest Snapshot", kpis.skus_latest_snapshot],
    ["Low/Critical Items", kpis.low_critical_items],
    ["Transfer Opportunities", kpis.transfer_opportunities],
  ];
  rows.forEach(([label, value]) => {
    const div = document.createElement("div");
    div.className = "kpi";
    div.innerHTML = `<div class="kpi-label">${label}</div><div class="kpi-value">${value}</div>`;
    target.appendChild(div);
  });
}

function renderTable(containerId, rows, preferredColumns = []) {
  const target = document.getElementById(containerId);
  if (!rows || rows.length === 0) {
    target.innerHTML = "<p>No rows.</p>";
    return;
  }
  const columns = preferredColumns.length ? preferredColumns : Object.keys(rows[0]);
  const header = `<tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const body = rows
    .map((row) => {
      return `<tr>${columns
        .map((c) => `<td>${row[c] === undefined || row[c] === null ? "-" : row[c]}</td>`)
        .join("")}</tr>`;
    })
    .join("");
  target.innerHTML = `<table>${header}${body}</table>`;
}

const forecastUiState = {
  distForecast: { expanded: "" },
  ownerForecast: { expanded: "" },
};

function normalize(value) {
  return String(value ?? "").trim().toLowerCase();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMedicine(value) {
  return String(value ?? "").replaceAll("_", " ");
}

function parseCurrency(value) {
  const cleaned = String(value ?? "").replace(/[^0-9.-]/g, "");
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : 0;
}

function estimateDaysUntilStockout(currentStock, combinedDemand) {
  const stock = Number(currentStock);
  const combined = Number(combinedDemand);
  if (!Number.isFinite(stock) || stock <= 0) return 0;
  if (!Number.isFinite(combined) || combined <= 0) return 999;
  const daily = combined / 3;
  if (daily <= 0) return 999;
  return Math.max(0, Math.floor(stock / daily));
}

function deriveStockStatus(predictedStatus, daysUntilStockout) {
  const status = normalize(predictedStatus);
  if (status.includes("critical") || status.includes("low") || daysUntilStockout <= 7) return "low";
  if (daysUntilStockout >= 20) return "surplus";
  return "optimal";
}

function deriveTrend(day1, day3) {
  const d1 = Number(day1);
  const d3 = Number(day3);
  if (!Number.isFinite(d1) || !Number.isFinite(d3)) return "stable";
  const diff = d3 - d1;
  if (diff > 1) return "rising";
  if (diff < -1) return "falling";
  return "stable";
}

function confidenceRange(day1, day2, day3) {
  const values = [Number(day1), Number(day2), Number(day3)].filter(Number.isFinite);
  if (!values.length) return { low: 0, high: 0 };
  const avg = values.reduce((acc, v) => acc + v, 0) / values.length;
  const spread = Math.max(1, (Math.max(...values) - Math.min(...values)) / 2);
  return {
    low: Number((avg - spread).toFixed(2)),
    high: Number((avg + spread).toFixed(2)),
  };
}

function seasonalFactor(combinedDemand) {
  const avg = Number(combinedDemand) / 3;
  if (avg >= 30) return "high";
  if (avg <= 12) return "low";
  return "normal";
}

function sparkBarsHtml(row) {
  const values = [Number(row.day_1_demand), Number(row.day_2_demand), Number(row.day_3_demand)].map((v) =>
    Number.isFinite(v) ? v : 0
  );
  const max = Math.max(1, ...values);
  const bars = values
    .map((v, idx) => {
      const height = Math.max(8, Math.round((v / max) * 28));
      return `<span class="forecast-spark-bar" style="height:${height}px" title="Day ${idx + 1}: ${v.toFixed(
        2
      )} units"></span>`;
    })
    .join("");
  return `<div class="forecast-spark-wrap"><div class="forecast-spark">${bars}</div><span class="forecast-total">${Number(
    row.combined_3_day_demand || 0
  ).toFixed(2)}</span></div>`;
}

function renderForecastDashboard(containerId, rows, options = {}) {
  const target = document.getElementById(containerId);
  const ui = forecastUiState[containerId] || { expanded: "" };
  const showStore = !!options.showStore;
  const filtered = [...rows].sort(
    (a, b) => Number(a.days_until_stockout || 999) - Number(b.days_until_stockout || 999)
  );

  const summary = {
    total: rows.length,
    low: rows.filter((r) => normalize(r.stock_status) === "low").length,
    recalled: rows.filter((r) => normalize(r.recall_status) === "recalled").length,
    reorder: rows.filter((r) => !!r.reorder_recommended).length,
  };

  const tableHeader = `
    <tr>
      ${showStore ? "<th>Store</th>" : ""}
      <th>Medicine</th>
      <th>Current Stock</th>
      <th>Days Until Stockout</th>
      <th>3-Day Demand Forecast</th>
    </tr>
  `;

  const tableRows = filtered
    .map((row, idx) => {
      const key = `${row.store_name}||${row.medicine_name}||${idx}`;
      const recalled = normalize(row.recall_status) === "recalled";
      const urgent = Number(row.days_until_stockout || 999) <= 7;
      const rowClass = recalled ? "forecast-row recalled" : urgent ? "forecast-row urgent" : "forecast-row";
      const stockDot =
        normalize(row.stock_status) === "low"
          ? "dot-low"
          : normalize(row.stock_status) === "optimal"
          ? "dot-optimal"
          : "dot-surplus";
      const stockoutClass =
        Number(row.days_until_stockout || 999) <= 7
          ? "stockout-critical"
          : Number(row.days_until_stockout || 999) <= 14
          ? "stockout-warning"
          : "stockout-safe";
      const insight = escapeHtml(
        `Demand is ${normalize(row.trend)} with ${normalize(row.seasonal_factor)} season pressure. Recommend ordering ${
          row.reorder_quantity || 0
        } units within ${Number(row.days_until_stockout || 999) <= 7 ? 3 : 7} days.`
      );
      const expanded = ui.expanded === key;
      return `
        <tr class="${rowClass}" data-row-key="${escapeHtml(key)}">
          ${showStore ? `<td>${escapeHtml(row.store_name)}</td>` : ""}
          <td>${escapeHtml(formatMedicine(row.medicine_name))}</td>
          <td><div class="stock-cell"><span class="stock-dot ${stockDot}"></span>${escapeHtml(
            row.current_stock
          )}</div></td>
          <td><span class="${stockoutClass}">${urgent ? "⚠ " : ""}${escapeHtml(row.days_until_stockout)}</span></td>
          <td>${sparkBarsHtml(row)}</td>
        </tr>
        ${
          expanded
            ? `<tr class="forecast-expand"><td colspan="${showStore ? 5 : 4}">
                <div class="forecast-expand-grid">
                  <p><strong>CDC Flu Index:</strong> ${Number(row.cdc_flu_index || 0).toFixed(1)}</p>
                  <p><strong>FDA Shortage Flag:</strong> ${row.fda_shortage_flag ? "Active" : "Not active"}</p>
                  <p><strong>Confidence Range:</strong> ${Number(row.confidence_low || 0).toFixed(2)} to ${Number(
                row.confidence_high || 0
              ).toFixed(2)}</p>
                  <p><strong>AI Insight:</strong> ${insight}</p>
                </div>
              </td></tr>`
            : ""
        }
      `;
    })
    .join("");

  const mobileCards = filtered
    .map((row, idx) => {
      const key = `${row.store_name}||${row.medicine_name}||${idx}`;
      const expanded = ui.expanded === key;
      const rowClass =
        normalize(row.recall_status) === "recalled" ? "forecast-card recalled" : "forecast-card";
      return `
      <div class="${rowClass}" data-row-key="${escapeHtml(key)}">
        <div class="forecast-card-top">
          <div>
            <p class="forecast-card-med">${escapeHtml(formatMedicine(row.medicine_name))}</p>
            ${showStore ? `<p class="forecast-card-store">${escapeHtml(row.store_name)}</p>` : ""}
          </div>
          <span class="${Number(row.days_until_stockout || 999) <= 7 ? "stockout-critical" : "stockout-warning"}">${escapeHtml(
        row.days_until_stockout
      )} days</span>
        </div>
        <p>Stock: ${escapeHtml(row.current_stock)} | Trend: ${escapeHtml(row.trend)}</p>
        <p>3-day demand: ${Number(row.combined_3_day_demand || 0).toFixed(2)}</p>
        ${
          expanded
            ? `<div class="forecast-card-expand">
                <p>CDC Flu Index: ${Number(row.cdc_flu_index || 0).toFixed(1)}</p>
                <p>Confidence: ${Number(row.confidence_low || 0).toFixed(2)} to ${Number(
                row.confidence_high || 0
              ).toFixed(2)}</p>
              </div>`
            : ""
        }
      </div>`;
    })
    .join("");

  target.innerHTML = `
    <section class="forecast-shell">
      <div class="forecast-summary">
        <div class="forecast-summary-card"><p>Total drugs tracked</p><strong>${summary.total}</strong></div>
        <div class="forecast-summary-card"><p>Low stock</p><strong>${summary.low}</strong></div>
        <div class="forecast-summary-card"><p>Active recalls</p><strong>${summary.recalled}</strong></div>
        <div class="forecast-summary-card"><p>Need reorder</p><strong>${summary.reorder}</strong></div>
      </div>
      <div class="forecast-table-wrap">
        <table class="forecast-table">
          <thead>${tableHeader}</thead>
          <tbody>${tableRows || `<tr><td colspan="${showStore ? 5 : 4}">No matching rows.</td></tr>`}</tbody>
        </table>
      </div>
      <div class="forecast-cards">${mobileCards || "<p>No matching rows.</p>"}</div>
    </section>
  `;

  target.querySelectorAll("[data-row-key]").forEach((rowEl) => {
    rowEl.addEventListener("click", () => {
      const key = rowEl.getAttribute("data-row-key") || "";
      ui.expanded = ui.expanded === key ? "" : key;
      renderForecastDashboard(containerId, rows, options);
    });
  });
}

function buildCompactForecastRows(inventoryRows, forecastRows, recommendationRows = [], lowStockRows = []) {
  // Build a compact, presentation-friendly forecast table:
  // one row per store+medicine with day1/day2/day3 and a combined 3-day total.
  const latestMap = new Map();
  inventoryRows.forEach((row) => {
    const key = `${row.store_name}||${row.medicine_name}`;
    latestMap.set(key, row);
  });

  const grouped = new Map();
  const recommendationMap = new Map();
  const lowStockSet = new Set(
    (lowStockRows || []).map((row) => `${row.store_name}||${row.medicine_name}`)
  );
  (recommendationRows || []).forEach((row) => {
    const key = `${row.store_name}||${row.medicine_name}`;
    recommendationMap.set(key, row);
  });
  forecastRows.forEach((row) => {
    const key = `${row.store_name}||${row.medicine_name}`;
    if (!grouped.has(key)) {
      grouped.set(key, {
        store_name: row.store_name,
        medicine_name: row.medicine_name,
        day_1_demand: null,
        day_2_demand: null,
        day_3_demand: null,
      });
    }
    const entry = grouped.get(key);
    const day = Number(row.forecast_day);
    const val = Number(row.predicted_demand);
    if (day === 1) entry.day_1_demand = Number.isFinite(val) ? val : null;
    if (day === 2) entry.day_2_demand = Number.isFinite(val) ? val : null;
    if (day === 3) entry.day_3_demand = Number.isFinite(val) ? val : null;
  });

  const out = [];
  grouped.forEach((entry, key) => {
    const latest = latestMap.get(key) || {};
    const d1 = entry.day_1_demand ?? 0;
    const d2 = entry.day_2_demand ?? 0;
    const d3 = entry.day_3_demand ?? 0;
    const stock = Number(latest.current_stock ?? 0);
    const daysUntilStockout = estimateDaysUntilStockout(stock, d1 + d2 + d3);
    const status = deriveStockStatus(latest.predicted_status, daysUntilStockout);
    const trend = deriveTrend(d1, d3);
    const confidence = confidenceRange(d1, d2, d3);
    const season = seasonalFactor(d1 + d2 + d3);
    const recommendation = recommendationMap.get(key) || {};
    const reason = `${recommendation.reason || ""} ${recommendation.action_details || ""}`.toLowerCase();
    const recallStatus = reason.includes("recall")
      ? "recalled"
      : lowStockSet.has(key)
      ? "shortage_warning"
      : "clear";
    const reorderRecommended = recommendation.final_decision === "Reorder Stock" || daysUntilStockout <= 10;
    const reorderQuantity = Number(recommendation.needed_units) || Math.max(0, Math.ceil((d1 + d2 + d3) - stock));
    const reorderCost = parseCurrency(recommendation.reorder_cost);
    const fluIndex = Number((1.4 + (season === "high" ? 1.3 : season === "normal" ? 0.8 : 0.4) + (trend === "rising" ? 0.7 : 0.2)).toFixed(1));
    const distributorAvailability =
      status === "low" && daysUntilStockout <= 4
        ? "out_of_stock"
        : status === "low"
        ? "low"
        : "in_stock";

    out.push({
      store_name: entry.store_name,
      medicine_name: entry.medicine_name,
      current_stock: latest.current_stock ?? "-",
      day_1_demand: entry.day_1_demand == null ? "-" : Number(entry.day_1_demand.toFixed(2)),
      day_2_demand: entry.day_2_demand == null ? "-" : Number(entry.day_2_demand.toFixed(2)),
      day_3_demand: entry.day_3_demand == null ? "-" : Number(entry.day_3_demand.toFixed(2)),
      combined_3_day_demand: Number((d1 + d2 + d3).toFixed(2)),
      days_until_stockout: daysUntilStockout,
      stock_status: status,
      trend,
      confidence_low: confidence.low,
      confidence_high: confidence.high,
      seasonal_factor: season,
      recall_status: recallStatus,
      reorder_recommended: reorderRecommended,
      reorder_quantity: reorderQuantity,
      estimated_reorder_cost: Number(reorderCost.toFixed(2)),
      distributor_availability: distributorAvailability,
      cdc_flu_index: fluIndex,
      fda_shortage_flag: recallStatus !== "clear" || daysUntilStockout <= 4,
    });
  });

  return out.sort((a, b) => {
    if (a.store_name === b.store_name) return a.medicine_name.localeCompare(b.medicine_name);
    return a.store_name.localeCompare(b.store_name);
  });
}

function uniqueValues(rows, key) {
  return [...new Set(rows.map((r) => r[key]).filter(Boolean))];
}

function fillSelectOptions(selectEl, values) {
  selectEl.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = true;
    selectEl.appendChild(option);
  });
}

function getSelectedMulti(selectEl) {
  return [...selectEl.options].filter((o) => o.selected).map((o) => o.value);
}

function renderDistributor() {
  const latest = state.data.latest_inventory || [];
  const forecast = state.data.forecast || [];
  const recs = state.data.recommendations || [];
  const low = state.data.low_stock || [];

  if (!distStoreFilter.options.length) {
    fillSelectOptions(distStoreFilter, uniqueValues(latest, "store_name"));
  }
  const selectedStores = getSelectedMulti(distStoreFilter);
  const include = (row) => selectedStores.includes(row.store_name);

  const storeFilteredInventory = latest.filter(include);
  const medicineChoices = ["All Medicines", ...uniqueValues(storeFilteredInventory, "medicine_name")];
  if (!distMedicineSelect.options.length) {
    distMedicineSelect.innerHTML = medicineChoices.map((m) => `<option value="${m}">${m}</option>`).join("");
  } else {
    const existing = [...distMedicineSelect.options].map((o) => o.value);
    if (existing.join("|") !== medicineChoices.join("|")) {
      distMedicineSelect.innerHTML = medicineChoices
        .map((m) => `<option value="${m}">${m}</option>`)
        .join("");
    }
  }
  const selectedMedicine = distMedicineSelect.value || "All Medicines";
  const medInclude = (row) =>
    include(row) && (selectedMedicine === "All Medicines" || row.medicine_name === selectedMedicine);

  const invView = latest.filter(medInclude);
  const forecastView = forecast.filter(medInclude);
  const recView = recs.filter(medInclude);
  const lowView = low.filter(medInclude);
  const compactForecast = buildCompactForecastRows(invView, forecastView, recView, lowView);

  window.__distInventoryView = invView;
  window.__distSelectedStores = selectedStores;
  window.__distSelectedMedicine = selectedMedicine;
  window.__distStoreFilteredInventory = storeFilteredInventory;

  renderTable("distInventory", invView);
  renderForecastDashboard("distForecast", compactForecast, { showStore: true });
  renderTable("distActionQueue", recView);
}

function renderOwner() {
  const latest = state.data.latest_inventory || [];
  const low = state.data.low_stock || [];
  const forecast = state.data.forecast || [];
  const recs = state.data.recommendations || [];

  if (!ownerStoreSelect.options.length) {
    const stores = uniqueValues(latest, "store_name");
    ownerStoreSelect.innerHTML = stores.map((s) => `<option value="${s}">${s}</option>`).join("");
  }
  const selectedStore = ownerStoreSelect.value;
  if (!selectedStore) return;

  const storeInventory = latest.filter((r) => r.store_name === selectedStore);
  const medicineOptions = ["All Medicines", ...uniqueValues(storeInventory, "medicine_name")];
  if (!ownerMedicineSelect.options.length) {
    ownerMedicineSelect.innerHTML = medicineOptions
      .map((m) => `<option value="${m}">${m}</option>`)
      .join("");
  } else {
    const existing = [...ownerMedicineSelect.options].map((o) => o.value);
    if (existing.join("|") !== medicineOptions.join("|")) {
      ownerMedicineSelect.innerHTML = medicineOptions
        .map((m) => `<option value="${m}">${m}</option>`)
        .join("");
    }
  }

  const selectedMedicine = ownerMedicineSelect.value || "All Medicines";
  const medFilter = (row) =>
    row.store_name === selectedStore &&
    (selectedMedicine === "All Medicines" || row.medicine_name === selectedMedicine);

  const ownerInventoryRows = latest.filter(medFilter);
  const ownerForecastRows = forecast.filter(medFilter);
  const ownerLowRows = low.filter(medFilter);
  const ownerRecs = recs.filter(medFilter);
  const ownerCompactForecast = buildCompactForecastRows(
    ownerInventoryRows,
    ownerForecastRows,
    ownerRecs,
    ownerLowRows
  );

  renderTable("ownerInventory", ownerInventoryRows);
  renderTable("ownerLow", ownerLowRows);
  renderForecastDashboard("ownerForecast", ownerCompactForecast, { showStore: false });

  window.__ownerSelectedStore = selectedStore;
  window.__ownerSelectedMedicine = selectedMedicine;
  window.__ownerInventoryRows = ownerInventoryRows;
  window.__ownerRecs = ownerRecs;
  renderAssistantWidget();
}

async function runAnalysis() {
  clearError();
  const fileInput = document.getElementById("csvFile");
  if (!fileInput.files.length) {
    showError("Upload a CSV file first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Analysis failed.");
      return;
    }
    state.analysisId = data.analysis_id;
    state.data = data;
    renderKpis(data.kpis);
    distStoreFilter.innerHTML = "";
    distMedicineSelect.innerHTML = "";
    ownerStoreSelect.innerHTML = "";
    ownerMedicineSelect.innerHTML = "";
    if (state.role === "Distributor") renderDistributor();
    if (state.role === "Store Owner") renderOwner();
  } catch (err) {
    showError(`Failed to run analysis: ${err.message}`);
  }
}

function renderFindings(containerId, findings) {
  const target = document.getElementById(containerId);
  if (!findings || findings.length === 0) {
    target.innerHTML = "<p>No findings generated.</p>";
    return;
  }
  const rows = findings.map((f) => {
    const evidence = (f.evidence || []).map((e) => `<li>${e}</li>`).join("");
    return `
      <tr>
        <td>${f.medicine_name}</td>
        <td>${f.finding}</td>
        <td>${f.action}</td>
        <td><ul>${evidence}</ul></td>
      </tr>
    `;
  });
  target.innerHTML = `
    <table>
      <tr><th>Medicine</th><th>Found</th><th>Recommended Action</th><th>Evidence</th></tr>
      ${rows.join("")}
    </table>
  `;
}

function renderNoAlertsMessage(targetId) {
  const target = document.getElementById(targetId);
  const now = new Date().toLocaleString();
  target.innerHTML = `
    <div class="chat-response">
      <p><strong>No active alerts found.</strong></p>
      <p>Last checked: ${now}</p>
    </div>
  `;
}

async function generateLiveFdaInsights(rows, targetId) {
  clearError();
  if (!state.analysisId) {
    showError("Run analysis before generating findings.");
    return;
  }
  const target = document.getElementById(targetId);
  target.innerHTML = `<div class="chat-response"><p>Fetching live FDA recalls and shortages...</p></div>`;

  if (!(rows || []).length) {
    renderNoAlertsMessage(targetId);
    return;
  }

  try {
    const role = targetId === "distInsights" ? "distributor" : "store_owner";
    const res = await fetch("/api/live-fda-insights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_id: state.analysisId,
        role,
        rows,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Live FDA request failed.");
    }
    const findings = data.findings || [];
    if (!findings.length) {
      renderNoAlertsMessage(targetId);
      return;
    }
    renderFindings(targetId, findings);

    const stamp = document.createElement("p");
    stamp.className = "text-xs";
    stamp.textContent = `Live FDA data updated: ${new Date(data.updated_at || Date.now()).toLocaleString()}`;
    target.prepend(stamp);
  } catch (err) {
    showError(`Live FDA fetch failed. (${err.message})`);
    target.innerHTML = `
      <div class="chat-response">
        <p><strong>Live FDA data unavailable.</strong></p>
        <p>Please retry in a few seconds.</p>
        <p><strong>Details:</strong> ${String(err.message || "").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</p>
      </div>
    `;
  }
}

async function generateDistributorInsights() {
  const rows = window.__distInventoryView || [];
  await generateLiveFdaInsights(rows, "distInsights");
}

async function generateOwnerInsights() {
  const rows = window.__ownerInventoryRows || [];
  await generateLiveFdaInsights(rows, "ownerInsights");
}

const assistantState = {
  open: false,
  activeTab: "ask",
  loadingByTab: { ask: false, recall: false, interaction: false, inventory: false },
  historyByTab: { ask: [], recall: [], interaction: [], inventory: [] },
  inputs: { askText: "", recallDrug: "", drug1: "", drug2: "", drug3: "", inventoryDrug: "" },
  showDrug3: false,
  hasAlertDot: false,
};

function assistantEscape(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function assistantAgo(ts) {
  const secs = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins} minute${mins > 1 ? "s" : ""} ago`;
  const hrs = Math.floor(mins / 60);
  return `${hrs} hour${hrs > 1 ? "s" : ""} ago`;
}

function assistantPush(tab, msg) {
  const entry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: Date.now(),
    ...msg,
  };
  assistantState.historyByTab[tab].push(entry);
  return entry.id;
}

function assistantUpdateMessage(tab, id, updates) {
  const list = assistantState.historyByTab[tab] || [];
  const idx = list.findIndex((m) => m.id === id);
  if (idx < 0) return;
  list[idx] = { ...list[idx], ...updates };
}

function assistantSetLoading(tab, value) {
  assistantState.loadingByTab[tab] = value;
  renderAssistantWidget();
}

function assistantToneClass(tone) {
  if (tone === "danger") return "pp-assistant-card-danger";
  if (tone === "warning") return "pp-assistant-card-warning";
  if (tone === "success") return "pp-assistant-card-success";
  return "pp-assistant-card-neutral";
}

async function assistantClaude(systemPrompt, userPrompt) {
  if (!GEMINI_KEY) {
    throw new Error("Missing Gemini API key.");
  }
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${encodeURIComponent(
      GEMINI_KEY
    )}`,
    {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      systemInstruction: {
        parts: [{ text: systemPrompt }],
      },
      contents: [{ role: "user", parts: [{ text: userPrompt }] }],
      generationConfig: {
        maxOutputTokens: 1000,
      },
    }),
  }
  );
  const payload = await res.json();
  if (!res.ok) throw new Error(payload?.error?.message || "Gemini request failed");
  const parts = payload?.candidates?.[0]?.content?.parts || [];
  return parts.map((part) => part?.text || "").filter(Boolean).join("\n").trim();
}

const AGENTIC_TOOL_DEFS = [
  {
    function_declarations: [
      {
        name: "check_fda_recall",
        description: "Check if a drug has an active FDA recall",
        parameters: {
          type: "OBJECT",
          properties: {
            drug_name: { type: "STRING" },
          },
          required: ["drug_name"],
        },
      },
      {
        name: "check_drug_interaction",
        description: "Check interaction between two drugs using NIH RxNorm",
        parameters: {
          type: "OBJECT",
          properties: {
            drug1: { type: "STRING" },
            drug2: { type: "STRING" },
          },
          required: ["drug1", "drug2"],
        },
      },
      {
        name: "check_inventory",
        description: "Check stock levels and ML forecast for a drug",
        parameters: {
          type: "OBJECT",
          properties: {
            drug_name: { type: "STRING" },
          },
          required: ["drug_name"],
        },
      },
    ],
  },
];

function assistantToolLabel(toolName) {
  const labels = {
    check_fda_recall: "FDA Recall Database",
    check_drug_interaction: "NIH RxNorm API",
    check_inventory: "ML Inventory Forecast",
  };
  return labels[toolName] || toolName;
}

function assistantToolBadge(toolName) {
  const badges = {
    check_fda_recall: "🔴 FDA Recall Check",
    check_drug_interaction: "⚕️ NIH RxNorm",
    check_inventory: "📦 ML Forecast",
  };
  return badges[toolName] || toolName;
}

function assistantNormalizeQuery(value) {
  return String(value || "")
    .toLowerCase()
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim();
}

function assistantFindInventoryMatch(drugName) {
  const q = assistantNormalizeQuery(drugName);
  const latestRows = state.data?.latest_inventory || [];
  const forecastRows = buildCompactForecastRows(
    latestRows,
    state.data?.forecast || [],
    state.data?.recommendations || [],
    state.data?.low_stock || []
  );
  const forecastMatch = forecastRows.find((d) => {
    const name = assistantNormalizeQuery(d.medicine_name);
    return name.includes(q) || q.includes(name);
  });
  const latestMatch = latestRows.find((d) => {
    const name = assistantNormalizeQuery(d.medicine_name);
    return name.includes(q) || q.includes(name);
  });
  return (
    forecastMatch ||
    (latestMatch
      ? {
          medicine_name: latestMatch.medicine_name,
          current_stock: latestMatch.current_stock,
          combined_3_day_demand: Number(latestMatch.daily_demand || 0) * 3,
          days_until_stockout:
            Number(latestMatch.daily_demand || 0) > 0
              ? Math.floor(Number(latestMatch.current_stock || 0) / Number(latestMatch.daily_demand || 1))
              : 999,
          reorder_quantity: Math.max(
            0,
            Math.ceil(Number(latestMatch.daily_demand || 0) * 3 - Number(latestMatch.current_stock || 0))
          ),
        }
      : null)
  );
}

async function assistantToolRunRecall(drugName) {
  if (!state.analysisId) {
    return { ok: false, error: "Run analysis before using assistant tools." };
  }
  const res = await fetch("/api/live-fda-insights", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      analysis_id: state.analysisId,
      role: "store_owner",
      rows: [{ medicine_name: drugName, current_stock: 0 }],
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Live FDA request failed");
  const finding = (data.findings || [])[0] || {};
  return {
    ok: true,
    drug_name: drugName,
    recalled:
      String(finding.finding || "").toLowerCase().includes("recall match") || !!finding.recall_number,
    finding: finding.finding || "No direct recall match found.",
    recall_class: finding.recall_class || null,
    recall_reason: finding.recall_reason || null,
    affected_lots: finding.affected_lots || [],
    confidence: Number(finding.confidence || 0),
  };
}

async function assistantToolRunInteraction(drug1, drug2) {
  const res = await fetch("/api/rxnorm-interaction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ drug1, drug2, drug3: null }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "RxNorm interaction service unavailable");
  return {
    ok: true,
    drug1,
    drug2,
    interaction_found: !!data.interaction_found,
    severity: String(data.severity || "none"),
    description: String(data.description || "No major interaction found."),
    source: String(data.source_name || "NIH RxNorm"),
  };
}

async function assistantToolRunInventory(drugName) {
  const match = assistantFindInventoryMatch(drugName);
  if (!match) return { ok: true, found: false, drug_name: drugName };
  return {
    ok: true,
    found: true,
    drug_name: match.medicine_name,
    current_stock: Number(match.current_stock || 0),
    combined_3_day_demand: Number(match.combined_3_day_demand || 0),
    days_until_stockout: Number(match.days_until_stockout || 0),
    reorder_quantity: Number(match.reorder_quantity || 0),
  };
}

async function assistantClaudeToolRequest(systemPrompt, messages) {
  if (!GEMINI_KEY) throw new Error("Missing Gemini API key.");
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${encodeURIComponent(
      GEMINI_KEY
    )}`,
    {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      systemInstruction: {
        parts: [{ text: systemPrompt }],
      },
      tools: AGENTIC_TOOL_DEFS,
      contents: messages,
      generationConfig: {
        maxOutputTokens: 1000,
      },
    }),
  }
  );
  const payload = await res.json();
  if (!res.ok) throw new Error(payload?.error?.message || "Gemini tool-use request failed");
  return payload;
}

async function assistantRunAskAnything() {
  const userMessage = assistantState.inputs.askText.trim();
  if (!userMessage) return;
  assistantSetLoading("ask", true);
  assistantPush("ask", { type: "user", tone: "neutral", content: userMessage });
  assistantState.inputs.askText = "";
  const thinkingId = assistantPush("ask", {
    type: "loading",
    content: "Analyzing your question...",
  });
  renderAssistantWidget();

  try {
    assistantUpdateMessage("ask", thinkingId, { content: "Classifying intent and running tools..." });
    renderAssistantWidget();
    const res = await fetch("/api/ask-anything-hybrid", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_id: state.analysisId,
        question: userMessage,
        store_name: window.__ownerSelectedStore || "",
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Hybrid assistant request failed");

    const why = (data.citations || [])
      .slice(0, 3)
      .map((line, idx) => `${idx + 1}. ${line}`)
      .join("\n");
    const finalText = why ? `${data.answer}\n\nWhy this answer:\n${why}` : data.answer;
    const confidenceText = `Confidence: ${data.confidence_band || "Medium"} (${Math.round(
      Number(data.confidence_score || 0) * 100
    )}%)`;

    assistantState.historyByTab.ask = assistantState.historyByTab.ask.filter((m) => m.id !== thinkingId);
    assistantPush("ask", {
      type: "result",
      tone: "neutral",
      content: finalText || "No additional details available.",
      badges: [
        confidenceText,
        `Intent: ${String(data.intent || "general")}`,
        ...Array.from(
          new Set((data.tools_used || []).map((toolName) => assistantToolBadge(String(toolName || ""))))
        ),
      ],
    });
  } catch (err) {
    assistantState.historyByTab.ask = assistantState.historyByTab.ask.filter((m) => m.id !== thinkingId);
    assistantPush("ask", {
      type: "result",
      tone: "warning",
      content: `Ask Anything failed.\n${String(err.message || err)}`,
    });
  } finally {
    assistantSetLoading("ask", false);
  }
}

async function assistantRunRecallCheck() {
  const drug = assistantState.inputs.recallDrug.trim();
  if (!drug) return;
  assistantSetLoading("recall", true);
  assistantPush("recall", { type: "loading", content: `Checking FDA recall database for ${drug}...` });
  renderAssistantWidget();

  try {
    if (!state.analysisId) throw new Error("Run analysis before using assistant.");
    const res = await fetch("/api/live-fda-insights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_id: state.analysisId,
        role: "store_owner",
        rows: [{ medicine_name: drug, current_stock: 0 }],
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Live FDA request failed");
    const finding = (data.findings || [])[0];
    const confidence = Number(finding?.confidence || 0);
    const confidencePct = Math.round(confidence * 100);
    const recalled =
      String(finding?.finding || "").toLowerCase().includes("recall match") ||
      !!finding?.recall_number;

    const lots = Array.isArray(finding?.affected_lots) && finding.affected_lots.length
      ? finding.affected_lots.join(", ")
      : "All lots";
    const recallClass = finding?.recall_class || "Class II";
    const reason = finding?.recall_reason || "Reason not provided in latest FDA response.";
    const defaultContent = recalled
      ? `DO NOT DISPENSE — ${drug} is under active FDA ${recallClass} recall.
Reason: ${reason}
Affected lots: ${lots}
Tell the customer: "This item is under a current safety recall, so we cannot dispense it right now. We can offer a safe alternative."
Substitute: Ask pharmacist to provide a therapeutically appropriate alternative.`
      : `${drug} — No active FDA recall found.
Safe to dispense. Last checked: ${new Date().toLocaleDateString()}`;

    const system = `You are a pharmacy safety assistant. You will be given an FDA recall check result for a drug. Respond in this exact format only:

If recalled:
Line 1: DO NOT DISPENSE — [Drug Name] is under active FDA [Class] recall.
Line 2: Reason: [reason in one sentence]
Line 3: Affected lots: [lot numbers or "All lots"]
Line 4: Tell the customer: "[one sentence script]"
Line 5: Substitute: [suggest a common alternative]

If not recalled:
Line 1: [Drug Name] — No active FDA recall found.
Line 2: Safe to dispense. Last checked: [today's date]

Keep it under 6 lines. Be direct. No extra commentary.`;
    const user = `FDA recall check result: ${JSON.stringify(finding || null)}
Drug name entered: ${drug}
Today's date: ${new Date().toLocaleDateString()}`;
    let content = defaultContent;
    try {
      content = await assistantClaude(system, user);
    } catch (err) {
      content = defaultContent;
    }

    assistantState.hasAlertDot = assistantState.hasAlertDot || recalled;
    assistantPush("recall", {
      type: "result",
      tone: recalled ? "danger" : "success",
      content,
      badges: [
        "Source: FDA openFDA Enforcement Database",
        `Matched via TF-IDF NLP (confidence: ${confidencePct}%)`,
      ],
    });
  } catch (err) {
    assistantPush("recall", {
      type: "result",
      tone: "warning",
      content:
        `Live recall check failed for ${drug}. Please retry in a few seconds.`,
      badges: ["Source: FDA openFDA Enforcement Database (live request failed)"],
    });
  } finally {
    assistantState.historyByTab.recall = assistantState.historyByTab.recall.filter((m) => m.type !== "loading");
    assistantSetLoading("recall", false);
  }
}

async function assistantRunInteractionCheck() {
  const drug1 = assistantState.inputs.drug1.trim();
  const drug2 = assistantState.inputs.drug2.trim();
  const drug3 = assistantState.showDrug3 ? assistantState.inputs.drug3.trim() : "";
  if (!drug1 || !drug2) return;

  assistantSetLoading("interaction", true);
  assistantPush("interaction", {
    type: "loading",
    content: `Checking NIH RxNorm database for ${drug1} + ${drug2}${drug3 ? ` + ${drug3}` : ""}...`,
  });
  renderAssistantWidget();

  try {
    const res = await fetch("/api/rxnorm-interaction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drug1, drug2, drug3: drug3 || null }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "RxNorm interaction service unavailable");
    const severity = String(data.severity || "none").toUpperCase();
    const desc = String(data.description || "No major interaction found.");
    const sourceName = String(data.source_name || "NIH RxNorm ONCHigh Database");
    const rxcui1 = data?.rxcui?.drug1 || "-";
    const rxcui2 = data?.rxcui?.drug2 || "-";

    const defaultContent =
      severity === "HIGH" || severity === "MODERATE"
        ? `INTERACTION FOUND — ${drug1} + ${drug2}
Severity: ${severity}
Risk: ${desc}
Action: Do not dispense until pharmacist review.
Alternative: Ask pharmacist for safer substitute guidance.`
        : `No major interaction — ${drug1} + ${drug2}
Generally safe to dispense together.
Still advise customer to take as directed.`;

    const system = `You are a pharmacy safety assistant for new employees.
You will be given an NIH RxNorm drug interaction result.
Respond in this exact format only:

If interaction found (severity high or moderate):
Line 1: INTERACTION FOUND — [Drug1] + [Drug2]
Line 2: Severity: [HIGH/MODERATE]
Line 3: Risk: [one sentence plain English explanation of danger]
Line 4: Action: [exactly what employee should do right now]
Line 5: Alternative: [safer substitute if applicable]

If low severity or no interaction:
Line 1: No major interaction — [Drug1] + [Drug2]
Line 2: Generally safe to dispense together.
Line 3: Still advise customer to take as directed.

Keep it under 6 lines. Plain English. No medical jargon.
Always end with a clear action for the employee.`;
    const user = `NIH RxNorm result: ${JSON.stringify(data)}
Drugs checked: ${drug1} and ${drug2}${drug3 ? ` and ${drug3}` : ""}`;
    let content = defaultContent;
    try {
      content = await assistantClaude(system, user);
    } catch (err) {
      content = defaultContent;
    }
    const tone = severity === "HIGH" ? "danger" : severity === "MODERATE" ? "warning" : "success";

    assistantPush("interaction", {
      type: "result",
      tone,
      content,
      badges: [
        `Source: ${sourceName}`,
        `rxcui: ${rxcui1} + ${rxcui2}`,
      ],
    });
  } catch (err) {
    assistantPush("interaction", {
      type: "result",
      tone: "warning",
      content:
        "NIH database temporarily unavailable.\nPlease consult the pharmacist on duty directly.",
      badges: ["Source: NIH RxNorm ONCHigh Database"],
    });
  } finally {
    assistantState.historyByTab.interaction = assistantState.historyByTab.interaction.filter(
      (m) => m.type !== "loading"
    );
    assistantSetLoading("interaction", false);
  }
}

async function assistantRunInventoryCheck() {
  const enteredDrugName = assistantState.inputs.inventoryDrug.trim();
  if (!enteredDrugName) return;
  assistantSetLoading("inventory", true);
  assistantPush("inventory", {
    type: "loading",
    content: `Checking inventory and ML forecast for ${enteredDrugName}...`,
  });
  renderAssistantWidget();

  try {
    const normalizeQuery = (value) =>
      String(value || "")
        .toLowerCase()
        .replaceAll("_", " ")
        .replace(/\s+/g, " ")
        .trim();

    const q = normalizeQuery(enteredDrugName);
    const latestRows = state.data?.latest_inventory || [];
    const forecastRows = buildCompactForecastRows(
      latestRows,
      state.data?.forecast || [],
      state.data?.recommendations || [],
      state.data?.low_stock || []
    );
    const forecastMatch = forecastRows.find((d) =>
      normalizeQuery(d.medicine_name).includes(q) || q.includes(normalizeQuery(d.medicine_name))
    );
    const latestMatch = latestRows.find((d) =>
      normalizeQuery(d.medicine_name).includes(q) || q.includes(normalizeQuery(d.medicine_name))
    );

    const match =
      forecastMatch ||
      (latestMatch
        ? {
            medicine_name: latestMatch.medicine_name,
            current_stock: latestMatch.current_stock,
            combined_3_day_demand: Number(latestMatch.daily_demand || 0) * 3,
            days_until_stockout:
              Number(latestMatch.daily_demand || 0) > 0
                ? Math.floor(Number(latestMatch.current_stock || 0) / Number(latestMatch.daily_demand || 1))
                : 999,
            reorder_quantity: Math.max(0, Math.ceil(Number(latestMatch.daily_demand || 0) * 3 - Number(latestMatch.current_stock || 0))),
          }
        : null);

    if (!match) {
      assistantPush("inventory", {
        type: "result",
        tone: "neutral",
        content: `${enteredDrugName} — Not found in current inventory\nCheck with your manager or look for alternatives`,
        badges: [
          "Random Forest Regression Forecast",
          `Updated: ${new Date().toLocaleString()}`,
        ],
      });
      return;
    }

    const healthy = Number(match.days_until_stockout || 999) > 7;
    const defaultContent = healthy
      ? `${match.medicine_name} — ${Number(match.current_stock || 0)} units in stock
Forecast: ~${Number(match.combined_3_day_demand || 0).toFixed(2)} units needed over next 3 days
Stock status: ${Number(match.days_until_stockout || 0)} days until stockout
Continue regular monitoring and dispense as directed.`
      : `${match.medicine_name} — ${Number(match.current_stock || 0)} units in stock (LOW)
Forecast: ~${Number(match.combined_3_day_demand || 0).toFixed(2)} units needed over next 3 days
Estimated stockout: ${Number(match.days_until_stockout || 0)} days
Notify manager to reorder ${Number(match.reorder_quantity || 0)} units immediately.`;

    const system = `You are a pharmacy inventory assistant for new employees.
You will be given ML forecast data for a drug.
Respond in this exact format only:

If drug found and stock healthy:
Line 1: [Drug Name] — [X] units in stock
Line 2: Forecast: ~[combined_3_day_demand] units needed over next 3 days
Line 3: Stock status: [X] days until stockout
Line 4: [one sentence action or reassurance]

If drug found and stock low (days_until_stockout <= 7):
Line 1: [Drug Name] — [X] units in stock (LOW)
Line 2: Forecast: ~[combined_3_day_demand] units needed over next 3 days
Line 3: Estimated stockout: [X] days
Line 4: Notify manager to reorder [reorder_quantity] units immediately

If drug not found in inventory:
Line 1: [Drug Name] — Not found in current inventory
Line 2: Check with your manager or look for alternatives

Keep it under 5 lines. Be specific with numbers.`;
    const user = `ML forecast data: ${JSON.stringify(match || null)}
Drug searched: ${enteredDrugName}`;

    let content = defaultContent;
    try {
      content = await assistantClaude(system, user);
    } catch (err) {
      content = defaultContent;
    }

    const tone = Number(match.days_until_stockout || 999) <= 7
      ? "warning"
      : "success";
    assistantPush("inventory", {
      type: "result",
      tone,
      content,
      badges: [
        "Random Forest Regression Forecast",
        `Updated: ${new Date().toLocaleString()}`,
      ],
    });
  } catch (err) {
    assistantPush("inventory", {
      type: "result",
      tone: "warning",
      content: `Inventory check failed for ${enteredDrugName}.\nPlease retry or contact your manager.`,
      badges: [
        "Random Forest Regression Forecast",
        `Updated: ${new Date().toLocaleString()}`,
      ],
    });
  } finally {
    assistantState.historyByTab.inventory = assistantState.historyByTab.inventory.filter(
      (m) => m.type !== "loading"
    );
    assistantSetLoading("inventory", false);
  }
}

function renderAssistantWidget() {
  const root = document.getElementById("ownerAssistantWidget");
  if (!root) return;
  if (state.role !== "Store Owner") {
    root.innerHTML = "";
    return;
  }

  const tab = assistantState.activeTab;
  const msgs = assistantState.historyByTab[tab] || [];
  const loading = assistantState.loadingByTab[tab];

  const tabsHtml = [
    ["ask", "Ask Anything"],
    ["recall", "Recall Check"],
    ["interaction", "Interaction Check"],
    ["inventory", "Inventory Check"],
  ]
    .map(
      ([id, label]) =>
        `<button class="pp-assistant-tab ${tab === id ? "active" : ""}" data-pp-tab="${id}">${label}</button>`
    )
    .join("");

  const historyHtml = msgs.length
    ? msgs
        .map((m) => {
          if (m.type === "loading") {
            return `<div class="pp-assistant-typing"><p>${assistantEscape(m.content)}</p><div class="pp-dots"><span></span><span></span><span></span></div></div>`;
          }
          if (m.type === "user") {
            return `<div class="pp-assistant-item">
              <p class="pp-assistant-time">${assistantAgo(m.timestamp)}</p>
              <div class="pp-assistant-card pp-assistant-card-user"><pre>${assistantEscape(m.content)}</pre></div>
            </div>`;
          }
          return `<div class="pp-assistant-item">
            <p class="pp-assistant-time">${assistantAgo(m.timestamp)}</p>
            <div class="pp-assistant-card ${assistantToneClass(m.tone)}"><pre>${assistantEscape(
            m.content
          )}</pre></div>
            ${
              (m.badges || []).length
                ? `<div class="pp-assistant-badges">${m.badges
                    .map((b) => `<span>${assistantEscape(b)}</span>`)
                    .join("")}</div>`
                : ""
            }
          </div>`;
        })
        .join("")
    : `<p class="pp-assistant-empty">${
        tab === "ask"
          ? "Ask anything about recalls, interactions, or stock."
          : tab === "recall"
          ? "Enter a drug name to check FDA recalls"
          : tab === "interaction"
          ? "Enter two drugs to check for interactions"
          : "Enter a drug name to check inventory"
      }</p>`;

  let inputHtml = "";
  if (tab === "ask") {
    const showDemoChips = Boolean(window.PILLPILOT_DEMO_MODE);
    const demoChips = showDemoChips
      ? `<div class="pp-assistant-chips">
          <button class="pp-assistant-chip" data-pp-ask-chip="Customer wants Children's Ibuprofen">Customer wants Children's Ibuprofen</button>
          <button class="pp-assistant-chip" data-pp-ask-chip="Customer takes warfarin and wants ibuprofen">Customer takes warfarin and wants ibuprofen</button>
          <button class="pp-assistant-chip" data-pp-ask-chip="Do we have amoxicillin in stock?">Do we have amoxicillin in stock?</button>
        </div>`
      : "";
    inputHtml = `
      ${demoChips}
      <label>Ask Anything</label>
      <input id="ppAskInput" value="${assistantEscape(
        assistantState.inputs.askText
      )}" placeholder="e.g. Customer wants ibuprofen and takes warfarin" ${loading ? "disabled" : ""} />
      <button id="ppAskSubmit" class="pp-assistant-submit primary" ${
        loading ? "disabled" : ""
      }>Send</button>
    `;
  } else if (tab === "recall") {
    inputHtml = `
      <label>Drug Name</label>
      <input id="ppRecallDrug" value="${assistantEscape(
        assistantState.inputs.recallDrug
      )}" placeholder="e.g. Children's Ibuprofen" ${loading ? "disabled" : ""} />
      <button id="ppRecallSubmit" class="pp-assistant-submit danger" ${
        loading ? "disabled" : ""
      }>Check FDA Recalls</button>
    `;
  } else if (tab === "interaction") {
    inputHtml = `
      <div class="pp-assistant-grid2">
        <div><label>Drug 1</label><input id="ppDrug1" value="${assistantEscape(
          assistantState.inputs.drug1
        )}" placeholder="e.g. Warfarin" ${loading ? "disabled" : ""} /></div>
        <div><label>Drug 2</label><input id="ppDrug2" value="${assistantEscape(
          assistantState.inputs.drug2
        )}" placeholder="e.g. Ibuprofen" ${loading ? "disabled" : ""} /></div>
      </div>
      ${
        assistantState.showDrug3
          ? `<div><label>Drug 3</label><input id="ppDrug3" value="${assistantEscape(
              assistantState.inputs.drug3
            )}" placeholder="Optional 3rd drug" ${loading ? "disabled" : ""} /></div>`
          : `<button id="ppAddDrug3" class="pp-assistant-link">Add 3rd drug +</button>`
      }
      <button id="ppInteractionSubmit" class="pp-assistant-submit primary" ${
        loading ? "disabled" : ""
      }>Check Interaction</button>
    `;
  } else {
    inputHtml = `
      <label>Drug Name</label>
      <input id="ppInventoryDrug" value="${assistantEscape(
        assistantState.inputs.inventoryDrug
      )}" placeholder="e.g. Amoxicillin 250mg" ${loading ? "disabled" : ""} />
      <button id="ppInventorySubmit" class="pp-assistant-submit success" ${
        loading ? "disabled" : ""
      }>Check Inventory</button>
    `;
  }

  root.innerHTML = `<div class="pp-assistant-panel pp-assistant-inline">
        <div class="pp-assistant-header">
          <div><p>Pharmacy Assistant</p><span><i></i>Online</span></div>
        </div>
        <div class="pp-assistant-tabs">${tabsHtml}</div>
        <div class="pp-assistant-input">${inputHtml}</div>
        <div class="pp-assistant-history">${historyHtml}${
          loading ? '<div class="pp-assistant-typing"><p>Checking...</p><div class="pp-dots"><span></span><span></span><span></span></div></div>' : ""
        }</div>
      </div>`;
  root.querySelectorAll("[data-pp-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      assistantState.activeTab = btn.getAttribute("data-pp-tab") || "ask";
      renderAssistantWidget();
    });
  });

  const askInput = document.getElementById("ppAskInput");
  if (askInput) {
    askInput.addEventListener("input", (e) => {
      assistantState.inputs.askText = e.target.value;
    });
    askInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        assistantRunAskAnything();
      }
    });
  }
  const askSubmit = document.getElementById("ppAskSubmit");
  if (askSubmit) askSubmit.addEventListener("click", assistantRunAskAnything);
  root.querySelectorAll("[data-pp-ask-chip]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-pp-ask-chip") || "";
      assistantState.inputs.askText = text;
      assistantRunAskAnything();
    });
  });

  const recallDrug = document.getElementById("ppRecallDrug");
  if (recallDrug) {
    recallDrug.addEventListener("input", (e) => {
      assistantState.inputs.recallDrug = e.target.value;
    });
  }
  const recallSubmit = document.getElementById("ppRecallSubmit");
  if (recallSubmit) recallSubmit.addEventListener("click", assistantRunRecallCheck);

  const drug1 = document.getElementById("ppDrug1");
  if (drug1) drug1.addEventListener("input", (e) => (assistantState.inputs.drug1 = e.target.value));
  const drug2 = document.getElementById("ppDrug2");
  if (drug2) drug2.addEventListener("input", (e) => (assistantState.inputs.drug2 = e.target.value));
  const drug3 = document.getElementById("ppDrug3");
  if (drug3) drug3.addEventListener("input", (e) => (assistantState.inputs.drug3 = e.target.value));
  const addDrug3 = document.getElementById("ppAddDrug3");
  if (addDrug3) {
    addDrug3.addEventListener("click", () => {
      assistantState.showDrug3 = true;
      renderAssistantWidget();
    });
  }
  const interactionSubmit = document.getElementById("ppInteractionSubmit");
  if (interactionSubmit) interactionSubmit.addEventListener("click", assistantRunInteractionCheck);

  const inventoryDrug = document.getElementById("ppInventoryDrug");
  if (inventoryDrug) {
    inventoryDrug.addEventListener("input", (e) => (assistantState.inputs.inventoryDrug = e.target.value));
  }
  const inventorySubmit = document.getElementById("ppInventorySubmit");
  if (inventorySubmit) inventorySubmit.addEventListener("click", assistantRunInventoryCheck);
}

document.getElementById("goDistributor").addEventListener("click", () => switchRole("Distributor"));
document.getElementById("goOwner").addEventListener("click", () => switchRole("Store Owner"));
document.getElementById("backBtn").addEventListener("click", goBack);
document.getElementById("runAnalysis").addEventListener("click", runAnalysis);
document.getElementById("distStoreFilter").addEventListener("change", renderDistributor);
document.getElementById("distMedicineSelect").addEventListener("change", renderDistributor);
document.getElementById("ownerStoreSelect").addEventListener("change", renderOwner);
document.getElementById("ownerMedicineSelect").addEventListener("change", renderOwner);
document.getElementById("distInsightsBtn").addEventListener("click", generateDistributorInsights);
document.getElementById("ownerInsightsBtn").addEventListener("click", generateOwnerInsights);
renderAssistantWidget();
