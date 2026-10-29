"use strict";

// "MEDIUM 33 -> HARD 23" reads better with a real arrow; applied once to every string in the data
const prettify = (v) => (typeof v === "string" ? v.replaceAll(" -> ", " → ")
  : Array.isArray(v) ? v.map(prettify)
  : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, prettify(x)])) : v);
const DATA = window.RESULTS && prettify(window.RESULTS);
const SVG_NS = "http://www.w3.org/2000/svg";
const COMPOUNDS = ["SOFT", "MEDIUM", "HARD"];
const GROUPS = ["VSC only", "Safety Car", "Wet", "Suspended"];
const renders = [];

// ---------- helpers ----------

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const compoundColor = (c) => css(`--${c.toLowerCase()}`);
const groupColor = (g) => css(["--accent", "--series-2", "--series-3", "--series-4"][GROUPS.indexOf(g)]);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const signed = (v, d = 2) => (v > 0 ? "+" : "") + fmt(v, d);
const thousands = (v) => Number(v).toLocaleString("en-US");

function h(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

function s(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

const scale = ([d0, d1], [r0, r1]) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

function ticks(min, max, count = 5) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => span / m <= count) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function svgRoot(el, height) {
  el.replaceChildren();
  const width = Math.max(el.clientWidth, 280);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  el.append(svg);
  return { svg, width };
}

// Bar with a 4px rounded data end, square at the baseline (horizontal, growing right or left)
function hBarPath(x0, x1, y, height, r = 4) {
  const left = Math.min(x0, x1), right = Math.max(x0, x1), w = right - left;
  const rr = Math.min(r, w, height / 2);
  if (x1 >= x0) {
    return `M${left},${y}H${right - rr}Q${right},${y} ${right},${y + rr}V${y + height - rr}Q${right},${y + height} ${right - rr},${y + height}H${left}Z`;
  }
  return `M${right},${y}H${left + rr}Q${left},${y} ${left},${y + rr}V${y + height - rr}Q${left},${y + height} ${left + rr},${y + height}H${right}Z`;
}

// ---------- tooltip ----------

const tooltip = document.getElementById("tooltip");

function showTip(event, title, rows) {
  tooltip.replaceChildren(h("div", { class: "t-title", text: title }));
  for (const row of rows) {
    const line = h("div", { class: "t-row" });
    if (row.color) line.append(h("span", { class: "t-key", style: `background:${row.color}` }));
    line.append(h("strong", { text: row.value }), h("span", { class: "t-name", text: row.name || "" }));
    tooltip.append(line);
  }
  const box = event.target.getBoundingClientRect ? event.target.getBoundingClientRect() : null;
  const x = event.clientX ?? (box ? box.right : 0);
  const y = event.clientY ?? (box ? box.top : 0);
  tooltip.classList.add("show");
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  tooltip.style.left = `${Math.min(x + 14, window.innerWidth - tw - 8)}px`;
  tooltip.style.top = `${Math.max(8, y - th - 12)}px`;
}

const hideTip = () => tooltip.classList.remove("show");

function hoverable(node, title, rows) {
  node.setAttribute("tabindex", "0");
  node.classList.add("mark");
  node.addEventListener("pointermove", (e) => showTip(e, title, rows()));
  node.addEventListener("focus", (e) => showTip(e, title, rows()));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("blur", hideTip);
}

// ---------- tables, legends, stats ----------

function table(el, columns, data, { open = false, highlight = () => false } = {}) {
  const head = h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "", text: c.label })));
  const body = data.map((row) => h("tr", { class: highlight(row) ? "highlight" : "" }, columns.map((c) =>
    h("td", { class: c.num ? "num" : "", text: c.format ? c.format(row[c.key], row) : String(row[c.key] ?? "–") }))));
  const t = h("div", { class: "table-wrap" }, h("table", {}, [h("thead", {}, head), h("tbody", {}, body)]));
  if (open) {
    el.replaceChildren(t);
    return;
  }
  el.replaceChildren(h("details", { class: "table-view" }, [h("summary", { text: "Show table" }), t]));
}

function legend(el, items, { shape = "rect", hidden, onToggle } = {}) {
  el.replaceChildren(...items.map((item) => {
    const key = h("span", { class: `key ${shape}`, style: `background:${item.color}` });
    if (!onToggle) return h("li", {}, [key, item.name]);
    const button = h("button", { type: "button", "aria-pressed": String(!hidden.has(item.name)) }, [key, item.name]);
    button.addEventListener("click", () => {
      hidden.has(item.name) ? hidden.delete(item.name) : hidden.add(item.name);
      onToggle();
    });
    return h("li", {}, button);
  }));
}

function stats(el, items) {
  el.replaceChildren(...items.map((item) => h("div", { class: `card stat ${item.hero ? "hero" : ""}` }, [
    h("div", { class: "label", text: item.label }),
    h("div", { class: "value", text: item.value }),
    item.note ? h("div", { class: `note ${item.noteClass || ""}`, text: item.note }) : null,
  ])));
}

// ---------- charts ----------

function barChart(el, items, { format = (v) => fmt(v, 3), reference, refLabel, labelWidth = 190, barHeight = 22 } = {}) {
  const gap = 10, top = reference !== undefined ? 24 : 8, bottom = 24;
  const height = top + items.length * (barHeight + gap) + bottom;
  const { svg, width } = svgRoot(el, height);
  const lw = Math.min(labelWidth, width * 0.42);
  const values = items.map((d) => d.value).concat(reference ?? []);
  const min = Math.min(0, ...values), max = Math.max(0, ...values);
  const x = scale([min, max * 1.12 || 1], [lw, width - 8]);
  for (const t of ticks(min, max * 1.12)) {
    svg.append(s("line", { class: "gridline", x1: x(t), x2: x(t), y1: top, y2: height - bottom }));
    svg.append(s("text", { x: x(t), y: height - 6, "text-anchor": "middle" }, String(t)));
  }
  items.forEach((d, i) => {
    const y = top + i * (barHeight + gap);
    svg.append(s("text", { x: lw - 8, y: y + barHeight / 2 + 4, "text-anchor": "end", class: d.strong ? "label-strong" : "label-2" }, d.label));
    const bar = s("path", { d: hBarPath(x(0), x(d.value), y, barHeight), fill: d.color });
    hoverable(bar, d.label, () => d.tip || [{ value: format(d.value), color: d.color }]);
    svg.append(bar);
    const tip = x(d.value);
    svg.append(s("text", { x: d.value >= 0 ? tip + 6 : tip - 6, y: y + barHeight / 2 + 4, "text-anchor": d.value >= 0 ? "start" : "end", class: "label-strong" }, format(d.value)));
  });
  svg.append(s("line", { class: "baseline", x1: x(0), x2: x(0), y1: top, y2: height - bottom }));
  if (reference !== undefined) {
    svg.append(s("line", { x1: x(reference), x2: x(reference), y1: top - 6, y2: height - bottom, stroke: css("--ink-2"), "stroke-width": 1, "stroke-dasharray": "4 3" }));
    if (refLabel) svg.append(s("text", { x: x(reference), y: top - 10, "text-anchor": "middle", class: "label-2" }, refLabel));
  }
}

function lineChart(el, series, { height = 300, xLabel, yLabel, xFormat = (v) => v, yFormat = (v) => fmt(v), band, vlines = [], tipTitle } = {}) {
  const { svg, width } = svgRoot(el, height);
  const m = { top: 12, right: 16, bottom: 40, left: 68 };
  const all = series.flatMap((d) => d.points);
  if (!all.length) return;
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  const x = scale([Math.min(...xs), Math.max(...xs)], [m.left, width - m.right]);
  const yMin = Math.min(...ys), yMax = Math.max(...ys), pad = (yMax - yMin) * 0.08 || 0.1;
  const y = scale([yMin - pad, yMax + pad], [height - m.bottom, m.top]);
  if (band) svg.append(s("rect", { x: x(band[0]) - 3, width: x(band[1]) - x(band[0]) + 6, y: m.top, height: height - m.top - m.bottom, fill: css("--accent-wash") }));
  for (const t of ticks(yMin - pad, yMax + pad)) {
    svg.append(s("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(t), y2: y(t) }));
    svg.append(s("text", { x: m.left - 8, y: y(t) + 4, "text-anchor": "end" }, yFormat(t)));
  }
  for (const t of ticks(Math.min(...xs), Math.max(...xs), Math.max(3, Math.floor(width / 90)))) {
    svg.append(s("text", { x: x(t), y: height - m.bottom + 18, "text-anchor": "middle" }, xFormat(t)));
  }
  for (const v of vlines) svg.append(s("line", { x1: x(v.x), x2: x(v.x), y1: m.top, y2: height - m.bottom, stroke: v.color, "stroke-width": 1, "stroke-dasharray": "2 3" }));
  svg.append(s("line", { class: "baseline", x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom }));
  if (xLabel) svg.append(s("text", { x: (m.left + width - m.right) / 2, y: height - 4, "text-anchor": "middle", class: "label-2" }, xLabel));
  if (yLabel) svg.append(s("text", { x: -(m.top + height - m.bottom) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle", class: "label-2" }, yLabel));
  for (const d of series) {
    const path = d.points.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join("");
    svg.append(s("path", { d: path, fill: "none", stroke: d.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  }
  // Crosshair: snaps to the nearest x and lists every series there
  const xValues = [...new Set(xs)].sort((a, b) => a - b);
  const cross = s("line", { x1: 0, x2: 0, y1: m.top, y2: height - m.bottom, stroke: css("--ink-2"), "stroke-width": 1, visibility: "hidden" });
  const dots = series.map((d) => s("circle", { r: 4, fill: d.color, stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" }));
  svg.append(cross, ...dots);
  const hit = s("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent", tabindex: 0 });
  const at = (event) => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width;
    const xv = xValues.reduce((best, v) => (Math.abs(x(v) - px) < Math.abs(x(best) - px) ? v : best), xValues[0]);
    cross.setAttribute("x1", x(xv));
    cross.setAttribute("x2", x(xv));
    cross.setAttribute("visibility", "visible");
    const rows = [];
    series.forEach((d, i) => {
      const p = d.points.find((q) => q.x === xv);
      dots[i].setAttribute("visibility", p ? "visible" : "hidden");
      if (p) {
        dots[i].setAttribute("cx", x(p.x));
        dots[i].setAttribute("cy", y(p.y));
        rows.push({ color: d.color, name: d.name, value: yFormat(p.y) });
      }
    });
    showTip(event, tipTitle ? tipTitle(xv) : `${xLabel}: ${xFormat(xv)}`, rows);
  };
  hit.addEventListener("pointermove", at);
  hit.addEventListener("pointerleave", () => {
    hideTip();
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  });
  svg.append(hit);
}

function scatter(el, points, { height = 320, xLabel, yLabel, diagonal = false, tip, xDomain, yDomain } = {}) {
  const { svg, width } = svgRoot(el, height);
  const m = { top: 12, right: 16, bottom: 40, left: 60 };
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  let [x0, x1] = xDomain || [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = yDomain || [Math.min(...ys), Math.max(...ys)];
  if (diagonal) [x0, x1, y0, y1] = [Math.min(x0, y0), Math.max(x1, y1), Math.min(x0, y0), Math.max(x1, y1)];
  const px = (x1 - x0) * 0.05 || 1, py = (y1 - y0) * 0.05 || 1;
  const x = scale([x0 - px, x1 + px], [m.left, width - m.right]);
  const y = scale([y0 - py, y1 + py], [height - m.bottom, m.top]);
  for (const t of ticks(y0 - py, y1 + py)) {
    svg.append(s("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(t), y2: y(t) }));
    svg.append(s("text", { x: m.left - 8, y: y(t) + 4, "text-anchor": "end" }, String(t)));
  }
  for (const t of ticks(x0 - px, x1 + px, Math.max(3, Math.floor(width / 90)))) {
    svg.append(s("text", { x: x(t), y: height - m.bottom + 18, "text-anchor": "middle" }, String(t)));
  }
  svg.append(s("line", { class: "baseline", x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom }));
  if (diagonal) svg.append(s("line", { x1: x(x0 - px), y1: y(x0 - px), x2: x(x1 + px), y2: y(x1 + px), stroke: css("--axis"), "stroke-width": 1 }));
  if (xLabel) svg.append(s("text", { x: (m.left + width - m.right) / 2, y: height - 4, "text-anchor": "middle", class: "label-2" }, xLabel));
  if (yLabel) svg.append(s("text", { x: -(m.top + height - m.bottom) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle", class: "label-2" }, yLabel));
  const r = points.length > 300 ? 3 : 5;
  for (const p of points) {
    svg.append(s("circle", { cx: x(p.x), cy: y(p.y), r, fill: p.color, "fill-opacity": points.length > 300 ? 0.55 : 0.9, stroke: css("--surface"), "stroke-width": points.length > 300 ? 0.5 : 2 }));
  }
  // Nearest point within 24px, so small dots don't need to be hit exactly
  const ring = s("circle", { r: r + 3, fill: "none", stroke: css("--ink"), "stroke-width": 1.5, visibility: "hidden" });
  svg.append(ring);
  const hit = s("rect", { x: 0, y: 0, width, height, fill: "transparent" });
  hit.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const mx = ((event.clientX - box.left) / box.width) * width, my = ((event.clientY - box.top) / box.height) * height;
    let best = null, bestD = 24 * 24;
    for (const p of points) {
      const d = (x(p.x) - mx) ** 2 + (y(p.y) - my) ** 2;
      if (d < bestD) [best, bestD] = [p, d];
    }
    if (!best) {
      hideTip();
      ring.setAttribute("visibility", "hidden");
      return;
    }
    ring.setAttribute("cx", x(best.x));
    ring.setAttribute("cy", y(best.y));
    ring.setAttribute("visibility", "visible");
    const t = tip(best);
    showTip(event, t.title, t.rows);
  });
  hit.addEventListener("pointerleave", () => {
    hideTip();
    ring.setAttribute("visibility", "hidden");
  });
  svg.append(hit);
}

// ---------- sections ----------

function overview() {
  const d = DATA.dataset, models = DATA.models;
  const best = models.find((m) => m.model === "XGBoost (tuned, monotone)");
  const baseline = Math.min(...models.filter((m) => m.group === "baseline").map((m) => m.mae));
  const vsc = DATA.validation.summary.find((g) => g.group === "VSC only");
  const oneStop = DATA.strategies.filter((c) => c.best_2_stop_gap > 0).length;
  document.getElementById("overview-intro").textContent =
    `${d.races} races and ${thousands(d.clean_laps)} clean laps from OpenF1. Trained on 2024–2025, tested on ${DATA.test_year}.`;
  stats(document.getElementById("overview-stats"), [
    { label: "Lap-time error of the simulator's model", value: `${fmt(best.mae, 3)} s`, hero: true,
      note: `${fmt((1 - best.mae / baseline) * 100, 0)}% better than the best baseline (${fmt(baseline, 3)} s)`, noteClass: "delta-good" },
    { label: "Gap between top-5 drivers, VSC-only races", value: `±${fmt(vsc.gap_mae_s, 1)} s`, note: "over a whole race (mean absolute error)" },
    { label: "Finishing order, VSC-only races", value: fmt(vsc.spearman, 2), note: "Spearman correlation, predicted vs real" },
    { label: "Circuits where 1 stop beats 2", value: `${oneStop}/${DATA.strategies.length}`, note: "the model sees too little degradation for a 2nd stop" },
  ]);
}

function models() {
  const groupColors = { baseline: css("--neutral"), default: css("--accent"), tuned: css("--accent") };
  const toggle = document.getElementById("metric-toggle");
  let metric = "mae";
  const draw = () => {
    const ref = Math.min(...DATA.models.filter((m) => m.group === "baseline").map((m) => m[metric]));
    barChart(document.getElementById("models-chart"), DATA.models.map((m) => ({
      label: m.model, value: m[metric], color: groupColors[m.group], strong: m.model.includes("monotone"),
      tip: [{ value: `${fmt(m[metric], 3)} s`, name: metric.toUpperCase(), color: groupColors[m.group] },
        { value: signed(m[metric] - ref, 3), name: "vs best baseline" }],
    })), { reference: ref, refLabel: "best baseline", labelWidth: 210 });
  };
  toggle.addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    metric = button.dataset.metric;
    toggle.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    draw();
  });
  renders.push(draw);
  table(document.getElementById("models-table"), [
    { key: "model", label: "Model" }, { key: "group", label: "Kind" },
    { key: "mae", label: "MAE (s)", num: true, format: (v) => fmt(v, 3) }, { key: "rmse", label: "RMSE (s)", num: true, format: (v) => fmt(v, 3) },
  ], DATA.models, { highlight: (r) => r.model.includes("monotone") });

  const exp = DATA.circuit_experiment;
  renders.push(() => barChart(document.getElementById("circuit-chart"), exp.map((r) => ({
    label: r.features, value: r.mae, strong: r.features.includes("kept"),
    color: r.features.includes("kept") ? css("--accent") : css("--neutral"),
  })), { reference: exp[0].mae, refLabel: "no circuit features", labelWidth: 200 }));
  table(document.getElementById("circuit-table"), [
    { key: "features", label: "Features" }, { key: "mae", label: "Test MAE (s)", num: true, format: (v) => fmt(v, 4) },
  ], exp);
}

function explain() {
  const imp = DATA.shap.importance;
  renders.push(() => barChart(document.getElementById("shap-chart"), imp.filter((r) => r.mean_abs_shap_s > 0.0005).map((r) => ({
    label: r.feature.replace("team_name_", "team: "), value: r.mean_abs_shap_s, strong: r.feature === "compound",
    color: r.feature === "compound" ? css("--series-2") : css("--accent"),
  })), { format: (v) => fmt(v, 3), labelWidth: 170, barHeight: 16 }));
  table(document.getElementById("shap-table"), [
    { key: "feature", label: "Feature" }, { key: "mean_abs_shap_s", label: "Mean |SHAP| (s)", num: true, format: (v) => fmt(v, 4) },
  ], imp);

  const hiddenShap = new Set();
  const drawShap = () => {
    legend(document.getElementById("shap-legend"), COMPOUNDS.map((c) => ({ name: c, color: compoundColor(c) })),
      { shape: "rect", hidden: hiddenShap, onToggle: drawShap });
    scatter(document.getElementById("shap-scatter"),
      DATA.shap.tyre_life.filter((p) => !hiddenShap.has(p.compound)).map((p) => ({ x: p.tyre_life, y: p.shap, color: compoundColor(p.compound), d: p })),
      { xLabel: "Tyre life (laps)", yLabel: "SHAP of tyre_life (s)", xDomain: [0, 60], yDomain: [-0.7, 0.6],
        tip: (p) => ({ title: `${p.d.compound}, tyre life ${p.d.tyre_life}`, rows: [{ value: `${signed(p.d.shap, 3)} s`, name: "added to the lap", color: p.color }] }) });
  };
  renders.push(drawShap);

  const hiddenDeg = new Set();
  const deg = DATA.degradation;
  const drawDeg = () => {
    legend(document.getElementById("deg-legend"), COMPOUNDS.map((c) => ({ name: c, color: compoundColor(c) })),
      { shape: "line", hidden: hiddenDeg, onToggle: drawDeg });
    const visible = COMPOUNDS.filter((c) => !hiddenDeg.has(c));
    lineChart(document.getElementById("deg-chart"), visible.map((c) => ({
      name: c, color: compoundColor(c),
      points: deg.curves.filter((p) => p.compound === c).map((p) => ({ x: p.tyre_life, y: p.delta })),
    })), { xLabel: "Tyre life (laps)", yLabel: "Predicted lap delta (s)", yFormat: (v) => `${signed(v, 2)}`,
      vlines: visible.map((c) => ({ x: deg.limits[c], color: compoundColor(c) })), tipTitle: (v) => `Tyre life ${v} laps` });
  };
  renders.push(drawDeg);
  const wide = [];
  for (let t = 1; t <= 40; t++) {
    const row = { tyre_life: t };
    for (const c of COMPOUNDS) row[c] = deg.curves.find((p) => p.compound === c && p.tyre_life === t)?.delta;
    wide.push(row);
  }
  table(document.getElementById("deg-table"), [{ key: "tyre_life", label: "Tyre life", num: true },
    ...COMPOUNDS.map((c) => ({ key: c, label: `${c} (s)`, num: true, format: (v) => signed(v, 3) }))], wide);
}

function pitLoss() {
  const rows = [...DATA.pit_loss].sort((a, b) => a.pit_loss_b - b.pit_loss_b);
  const el = document.getElementById("pit-chart");
  const draw = () => {
    const a = css("--neutral"), b = css("--series-2");
    legend(document.getElementById("pit-legend"), [{ name: "A: pit-lane time", color: a }, { name: "B: race time lost (used)", color: b }], { shape: "rect" });
    const rowH = 22, top = 8, bottom = 30;
    const height = top + rows.length * rowH + bottom;
    const { svg, width } = svgRoot(el, height);
    const lw = Math.min(140, width * 0.35);
    const values = rows.flatMap((r) => [r.lane_duration, r.pit_loss_b]);
    const x = scale([Math.floor(Math.min(...values)) - 1, Math.ceil(Math.max(...values)) + 1], [lw, width - 12]);
    for (const t of ticks(Math.floor(Math.min(...values)) - 1, Math.ceil(Math.max(...values)) + 1, 6)) {
      svg.append(s("line", { class: "gridline", x1: x(t), x2: x(t), y1: top, y2: height - bottom }));
      svg.append(s("text", { x: x(t), y: height - 12, "text-anchor": "middle" }, `${t} s`));
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      svg.append(s("text", { x: lw - 8, y: y + 4, "text-anchor": "end", class: "label-2" }, r.circuit_short_name));
      svg.append(s("line", { x1: x(r.lane_duration), x2: x(r.pit_loss_b), y1: y, y2: y, stroke: css("--axis"), "stroke-width": 2 }));
      const g = s("g");
      g.append(s("rect", { x: lw, y: y - rowH / 2, width: width - lw, height: rowH, fill: "transparent" }));
      g.append(s("circle", { cx: x(r.lane_duration), cy: y, r: 5, fill: a, stroke: css("--surface"), "stroke-width": 2 }));
      g.append(s("circle", { cx: x(r.pit_loss_b), cy: y, r: 5, fill: b, stroke: css("--surface"), "stroke-width": 2 }));
      hoverable(g, `${r.circuit_short_name} (${r.n_stops} stops)`, () => [
        { value: `${fmt(r.pit_loss_b)} s`, name: "B: race time lost", color: b },
        { value: `${fmt(r.lane_duration)} s`, name: "A: pit-lane time", color: a },
        { value: `${fmt(r.stop_duration, 1)} s`, name: "stationary" },
      ]);
      svg.append(g);
    });
  };
  renders.push(draw);
  table(document.getElementById("pit-table"), [
    { key: "circuit_short_name", label: "Circuit" }, { key: "n_stops", label: "Stops", num: true },
    { key: "lane_duration", label: "A: lane (s)", num: true, format: (v) => fmt(v) },
    { key: "pit_loss_b", label: "B: lost (s)", num: true, format: (v) => fmt(v) },
    { key: "stop_duration", label: "Stationary (s)", num: true, format: (v) => fmt(v, 1) },
    { key: "pit_loss", label: "Used (s)", num: true, format: (v) => fmt(v) },
  ], rows);
}

function strategy() {
  document.getElementById("strategy-team").textContent = DATA.example.team;
  const select = document.getElementById("circuit-select");
  for (const c of DATA.strategies) select.append(h("option", { value: c.circuit, text: c.circuit }));
  select.value = DATA.example.circuit;
  const draw = () => {
    const c = DATA.strategies.find((d) => d.circuit === select.value);
    stats(document.getElementById("strategy-stats"), [
      { label: "Best strategy", value: c.best, note: `${thousands(c.n_strategies)} strategies simulated` },
      { label: "Race", value: `${c.total_laps} laps`, note: `track ${fmt(c.track_temperature, 1)} °C · pit loss ${fmt(c.pit_loss, 1)} s` },
      { label: "Pit window (within 1 s)", value: `laps ${c.window[0]}–${c.window[1]}`, note: `${c.pair.replace("-", " → ")} one-stop` },
      { label: "Best 2-stop loses", value: `+${fmt(c.best_2_stop_gap, 1)} s`, note: "against the best 1-stop" },
    ]);
    document.getElementById("window-title").textContent = `Pit window, ${c.pair.replace("-", " → ")}`;
    lineChart(document.getElementById("window-chart"), [{
      name: "vs best pit lap", color: css("--accent"), points: c.curve.map((p) => ({ x: p.pit_lap, y: p.gap })),
    }], { xLabel: "Pit lap", yLabel: "Seconds vs best pit lap", band: c.window, yFormat: (v) => `+${fmt(v, 1)} s`, height: 260 });
    table(document.getElementById("strategy-top"), [
      { key: "strategy", label: "Strategy" }, { key: "n_stops", label: "Stops", num: true },
      { key: "gap_to_best", label: "Gap (s)", num: true, format: (v) => `+${fmt(v)}` },
    ], c.top, { open: true, highlight: (r) => r.gap_to_best === 0 });
    const sorted = [...DATA.strategies].sort((a, b) => b.best_2_stop_gap - a.best_2_stop_gap);
    barChart(document.getElementById("stops-chart"), sorted.map((d) => ({
      label: d.circuit, value: d.best_2_stop_gap, strong: d.circuit === c.circuit,
      color: d.circuit === c.circuit ? css("--accent") : css("--neutral"),
      tip: [{ value: `+${fmt(d.best_2_stop_gap)} s`, name: "best 2-stop vs best 1-stop" }, { value: d.best, name: "best strategy" }],
    })), { format: (v) => `+${fmt(v, 1)}`, labelWidth: 150, barHeight: 14 });
  };
  select.addEventListener("change", draw);
  renders.push(draw);
  table(document.getElementById("stops-table"), [
    { key: "circuit", label: "Circuit" }, { key: "best", label: "Best strategy" },
    { key: "window", label: "Pit window", format: (v) => `${v[0]}–${v[1]}` },
    { key: "best_2_stop_gap", label: "2-stop loses (s)", num: true, format: (v) => `+${fmt(v)}` },
  ], DATA.strategies);
}

function validation() {
  const v = DATA.validation;
  const opt = v.optimal;
  const vsc = v.summary.find((g) => g.group === "VSC only");
  stats(document.getElementById("validation-stats"), [
    { label: "Drivers simulated", value: String(v.summary.reduce((a, g) => a + g.drivers, 0)), note: `${v.summary.reduce((a, g) => a + g.races, 0)} races of ${DATA.test_year}` },
    { label: "Race time error, VSC-only races", value: `${fmt(vsc.mae_pct, 1)}%`, note: `${fmt(vsc.mae_s, 0)} s, always too fast: neutralised laps are missing` },
    { label: "Same number of stops as the winner", value: `${opt.filter((r) => r.same_stops).length}/${opt.length}` },
    { label: "Same compounds as the winner", value: `${opt.filter((r) => r.same_compounds).length}/${opt.length}`, note: "the model can't tell compounds apart" },
  ]);
  const hidden = new Set();
  const draw = () => {
    legend(document.getElementById("validation-legend"), GROUPS.map((g) => ({ name: g, color: groupColor(g) })),
      { hidden, onToggle: draw });
    scatter(document.getElementById("validation-chart"),
      v.drivers.filter((d) => !hidden.has(d.group)).map((d) => ({ x: d.actual_gap, y: d.predicted_gap, color: groupColor(d.group), d })),
      { diagonal: true, xLabel: "Actual gap to race average (s)", yLabel: "Predicted gap (s)",
        tip: (p) => ({ title: `${p.d.circuit_short_name} · P${p.d.position} #${p.d.driver_number} ${p.d.team_name}`, rows: [
          { value: `${signed(p.d.actual_gap, 1)} s`, name: "actual gap" },
          { value: `${signed(p.d.predicted_gap, 1)} s`, name: "predicted gap", color: p.color },
          { value: p.d.strategy, name: "" },
        ] }) });
  };
  renders.push(draw);
  table(document.getElementById("validation-table"), [
    { key: "group", label: "Race type" }, { key: "races", label: "Races", num: true },
    { key: "mae_s", label: "Race time MAE", num: true, format: (x, r) => `${fmt(x, 0)} s (${fmt(r.mae_pct, 1)}%)` },
    { key: "gap_mae_s", label: "Gap MAE", num: true, format: (x) => `${fmt(x, 1)} s` },
    { key: "spearman", label: "Spearman", num: true, format: (x) => fmt(x, 2) },
  ], v.summary, { open: true });
  table(document.getElementById("optimal-table"), [
    { key: "circuit_short_name", label: "Race" }, { key: "group", label: "Type" },
    { key: "winner_strategy", label: "Winner" }, { key: "optimal_strategy", label: "Simulator's optimum" },
    { key: "same_stops", label: "Same stops", format: (x) => (x ? "✓ yes" : "– no") },
    { key: "predicted_gain_s", label: "Predicted gain", num: true, format: (x) => `${fmt(x, 1)} s` },
  ], opt, { open: true });
}

function safetyCar() {
  const sc = DATA.safety_car;
  const ratio = (t) => sc.neutral_pit_loss.find((r) => r.sc_type === t);
  document.getElementById("sc-intro").textContent =
    `${DATA.example.circuit}, ${DATA.example.team}: the best strategies run through the same ${thousands(sc.n_sims)} random races, ` +
    `with Safety Cars and VSCs drawn from the circuit's history. A stop under a neutralisation is cheaper, and the car brings its stop ` +
    `forward when one comes.`;
  stats(document.getElementById("sc-stats"), [
    { label: "Expected per race", value: `${fmt(sc.rates.SC, 2)} SC · ${fmt(sc.rates.VSC, 2)} VSC`, note: `${fmt(sc.neutralised_pct, 0)}% of random races neutralised` },
    { label: "Stop under Safety Car", value: `${fmt(ratio("SC").neutral_loss, 1)} s`, note: `${fmt(ratio("SC").ratio * 100, 0)}% of a green-flag stop (${ratio("SC").n_stops} stops)` },
    { label: "Stop under VSC", value: `${fmt(ratio("VSC").neutral_loss, 1)} s`, note: `${fmt(ratio("VSC").ratio * 100, 0)}% of a green-flag stop (${ratio("VSC").n_stops} stops)` },
    { label: "Best on average", value: sc.summary[0].strategy, note: `green-flag best: ${[...sc.summary].sort((a, b) => a.green_gap_s - b.green_gap_s)[0].strategy}` },
  ]);
  const draw = () => {
    const items = sc.regret, rowH = 34, top = 8, bottom = 30;
    const height = top + items.length * rowH + bottom;
    const { svg, width } = svgRoot(document.getElementById("sc-chart"), height);
    const lw = Math.min(170, width * 0.4);
    const max = Math.max(...items.map((d) => d.p95)) * 1.05;
    const x = scale([0, max], [lw, width - 12]);
    for (const t of ticks(0, max, 5)) {
      svg.append(s("line", { class: "gridline", x1: x(t), x2: x(t), y1: top, y2: height - bottom }));
      svg.append(s("text", { x: x(t), y: height - 12, "text-anchor": "middle" }, `${t} s`));
    }
    const color = css("--accent");
    items.forEach((d, i) => {
      const y = top + i * rowH + rowH / 2;
      svg.append(s("text", { x: lw - 8, y: y + 4, "text-anchor": "end", class: "label-2" }, d.strategy));
      const g = s("g");
      g.append(s("rect", { x: lw, y: y - rowH / 2, width: width - lw, height: rowH, fill: "transparent" }));
      g.append(s("line", { x1: x(d.p5), x2: x(d.p95), y1: y, y2: y, stroke: css("--axis"), "stroke-width": 2 }));
      g.append(s("rect", { x: x(d.p25), width: Math.max(2, x(d.p75) - x(d.p25)), y: y - 8, height: 16, rx: 4, fill: css("--accent-wash"), stroke: color, "stroke-width": 1 }));
      g.append(s("line", { x1: x(d.p50), x2: x(d.p50), y1: y - 8, y2: y + 8, stroke: color, "stroke-width": 2 }));
      g.append(s("circle", { cx: x(d.mean), cy: y, r: 4, fill: css("--series-2"), stroke: css("--surface"), "stroke-width": 2 }));
      hoverable(g, d.strategy, () => [
        { value: `${fmt(d.mean)} s`, name: "mean regret", color: css("--series-2") },
        { value: `${fmt(d.p50)} s`, name: "median", color },
        { value: `${fmt(d.p95)} s`, name: "95th percentile" },
      ]);
      svg.append(g);
    });
  };
  renders.push(draw);
  table(document.getElementById("sc-table"), [
    { key: "strategy", label: "Strategy" },
    { key: "green_gap_s", label: "Green-flag gap", num: true, format: (x) => `+${fmt(x)}` },
    { key: "expected_gap_s", label: "Expected gap", num: true, format: (x) => `+${fmt(x)}` },
    { key: "p90_regret_s", label: "p90 regret", num: true, format: (x) => fmt(x) },
    { key: "win_pct", label: "Win %", num: true, format: (x) => fmt(x, 1) },
  ], sc.summary.slice(0, 8), { open: true, highlight: (r) => r.expected_gap_s === 0 });
}

function mix(a, b, t) {
  const p = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const [ca, cb] = [p(a), p(b)];
  return `rgb(${ca.map((v, i) => Math.round(v + (cb[i] - v) * t)).join(",")})`;
}

function undercut() {
  const u = DATA.undercut;
  const oneLap = u.cells.filter((c) => c.laps_later === 1).map((c) => c.gain_s).sort((a, b) => a - b);
  const median = oneLap[Math.floor(oneLap.length / 2)];
  document.getElementById("undercut-intro").textContent =
    `Two identical ${DATA.example.team} cars at ${DATA.example.circuit}, both ${u.old} → ${u.new}. Car A pits first; car B stays out ` +
    `1–10 laps longer. Pitting one lap earlier gains about ${fmt(median, 2)} s, so the undercut works from up to ~${fmt(Math.max(...oneLap), 1)} s behind.`;
  const draw = () => {
    const el = document.getElementById("undercut-chart");
    const laps = [...new Set(u.cells.map((c) => c.pit_lap))].sort((a, b) => a - b);
    const waits = [...new Set(u.cells.map((c) => c.laps_later))].sort((a, b) => a - b);
    const m = { top: 8, right: 12, bottom: 56, left: 48 };
    const { svg, width } = svgRoot(el, 0);
    const cw = (width - m.left - m.right) / laps.length;
    const ch = Math.max(16, Math.min(26, cw * 1.2));
    const height = m.top + waits.length * ch + m.bottom;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const limit = Math.max(...u.cells.map((c) => Math.abs(c.gain_s)), 0.01);
    const [neg, mid, pos] = [css("--div-neg"), css("--div-mid"), css("--div-pos")];
    const color = (v) => (v >= 0 ? mix(mid, pos, v / limit) : mix(mid, neg, -v / limit));
    for (const c of u.cells) {
      const cell = s("rect", {
        x: m.left + laps.indexOf(c.pit_lap) * cw + 1, y: m.top + (waits.length - c.laps_later) * ch + 1,
        width: Math.max(1, cw - 2), height: ch - 2, rx: 2, fill: color(c.gain_s),
      });
      hoverable(cell, `A pits on lap ${c.pit_lap}, B ${c.laps_later} lap${c.laps_later > 1 ? "s" : ""} later`,
        () => [{ value: `${signed(c.gain_s)} s`, name: c.gain_s >= 0 ? "A gains (undercut)" : "B gains (overcut)" }]);
      svg.append(cell);
    }
    waits.forEach((w) => svg.append(s("text", { x: m.left - 8, y: m.top + (waits.length - w) * ch + ch / 2 + 4, "text-anchor": "end" }, String(w))));
    const every = Math.ceil(laps.length / Math.floor(width / 40));
    laps.forEach((l, i) => i % every === 0 && svg.append(s("text", { x: m.left + i * cw + cw / 2, y: m.top + waits.length * ch + 16, "text-anchor": "middle" }, String(l))));
    svg.append(s("text", { x: (m.left + width - m.right) / 2, y: m.top + waits.length * ch + 36, "text-anchor": "middle", class: "label-2" }, "Lap car A pits"));
    svg.append(s("text", { x: -(m.top + m.top + waits.length * ch) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle", class: "label-2" }, "Laps B stays out"));
    // Colour key
    const kx = width - m.right - 200, ky = height - 14;
    const grad = s("linearGradient", { id: "div-grad" });
    [[0, neg], [0.5, mid], [1, pos]].forEach(([o, c]) => grad.append(s("stop", { offset: o, "stop-color": c })));
    const defs = s("defs");
    defs.append(grad);
    svg.append(defs, s("rect", { x: kx + 40, y: ky - 9, width: 120, height: 10, rx: 3, fill: "url(#div-grad)" }));
    svg.append(s("text", { x: kx + 34, y: ky, "text-anchor": "end" }, `−${fmt(limit, 1)} s`));
    svg.append(s("text", { x: kx + 166, y: ky }, `+${fmt(limit, 1)} s`));
  };
  renders.push(draw);
}

function traffic() {
  const t = DATA.traffic;
  const real = t.ranking.find((r) => r.strategy.endsWith("(real)"));
  const freeBest = t.ranking.find((r) => r.free_rank === 1);
  document.getElementById("traffic-intro").textContent =
    `Our ${DATA.example.team} car races the cars on the lead lap of the latest ${DATA.example.circuit} race, each on its real strategy and ` +
    `own pace. Close behind a car it loses the dirty-air penalty and only passes if it is 0.7 s a lap faster.`;
  document.getElementById("traffic-callout").textContent =
    `With traffic, the strategy actually raced (${real.strategy.replace(" (real)", "")}) comes out ${fmt(freeBest.traffic_time - real.traffic_time, 1)} s ahead ` +
    `of the green-flag optimum (${freeBest.strategy}), which spends ${freeBest.laps_held} laps stuck behind another car and loses ` +
    `${fmt(freeBest.traffic_loss_s, 1)} s. Caveat: the real early stops were made under a Safety Car; here everyone pays the full pit loss.`;
  renders.push(() => barChart(document.getElementById("dirty-chart"), t.dirty_air.filter((r) => r.penalty_s > 0).map((r) => ({
    label: `${r.gap_bin} s behind`, value: r.penalty_s, color: css("--series-2"),
    tip: [{ value: `${fmt(r.penalty_s, 3)} s`, name: "lost per lap" }, { value: thousands(r.n_laps), name: "laps measured" }],
  })), { format: (v) => `${fmt(v, 2)} s`, labelWidth: 130 }));
  table(document.getElementById("dirty-table"), [
    { key: "gap_bin", label: "Gap to car ahead (s)" }, { key: "n_laps", label: "Laps", num: true, format: thousands },
    { key: "penalty_s", label: "Lost per lap (s)", num: true, format: (v) => fmt(v, 3) },
  ], t.dirty_air);

  const draw = () => {
    const free = css("--accent"), lost = css("--series-2");
    legend(document.getElementById("traffic-legend"), [{ name: "Gap without traffic", color: free }, { name: "Lost in traffic", color: lost }]);
    const items = t.ranking.slice(0, 10), barH = 18, gap = 10, top = 8, bottom = 28;
    const height = top + items.length * (barH + gap) + bottom;
    const { svg, width } = svgRoot(document.getElementById("traffic-chart"), height);
    const lw = Math.min(200, width * 0.45);
    const best = freeBest.free_time;
    const max = Math.max(...items.map((r) => r.traffic_time - best)) * 1.1;
    const x = scale([0, max], [lw, width - 12]);
    for (const tk of ticks(0, max, 5)) {
      svg.append(s("line", { class: "gridline", x1: x(tk), x2: x(tk), y1: top, y2: height - bottom }));
      svg.append(s("text", { x: x(tk), y: height - 10, "text-anchor": "middle" }, `${tk} s`));
    }
    items.forEach((r, i) => {
      const y = top + i * (barH + gap);
      const freeGap = r.free_time - best;
      svg.append(s("text", { x: lw - 8, y: y + barH / 2 + 4, "text-anchor": "end", class: r === real ? "label-strong" : "label-2" }, r.strategy));
      const g = s("g");
      if (freeGap > 0.01) g.append(s("rect", { x: x(0), y, width: Math.max(0, x(freeGap) - x(0) - 2), height: barH, fill: free }));
      g.append(s("path", { d: hBarPath(x(freeGap), x(freeGap + r.traffic_loss_s), y, barH), fill: lost }));
      hoverable(g, `${r.strategy} → P${r.position}`, () => [
        { value: `+${fmt(freeGap)} s`, name: "gap without traffic", color: free },
        { value: `${fmt(r.traffic_loss_s)} s`, name: `lost in traffic (${r.laps_held} laps held)`, color: lost },
        { value: `+${fmt(r.traffic_time - best)} s`, name: "total" },
      ]);
      svg.append(g);
    });
    svg.append(s("line", { class: "baseline", x1: x(0), x2: x(0), y1: top, y2: height - bottom }));
  };
  renders.push(draw);
  table(document.getElementById("traffic-table"), [
    { key: "strategy", label: "Strategy" }, { key: "free_rank", label: "Rank w/o traffic", num: true },
    { key: "traffic_loss_s", label: "Lost (s)", num: true, format: (v) => fmt(v) },
    { key: "laps_held", label: "Laps held", num: true }, { key: "position", label: "Finish", num: true, format: (v) => `P${v}` },
  ], t.ranking);
}

// ---------- page ----------

function renderAll() {
  hideTip();
  for (const draw of renders) draw();
}

function setupTheme() {
  const button = document.getElementById("theme-toggle");
  const stored = (() => { try { return localStorage.getItem("theme"); } catch { return null; } })();
  if (stored) document.documentElement.dataset.theme = stored;
  const isDark = () => document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const label = () => { button.textContent = isDark() ? "Light mode" : "Dark mode"; };
  label();
  button.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* private mode */ }
    label();
    renderAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { label(); renderAll(); });
}

function setupNav() {
  const links = [...document.querySelectorAll("nav.toc a")];
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${entry.target.id}`));
    }
  }, { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section").forEach((sec) => observer.observe(sec));
}

if (!DATA) {
  document.querySelector("main").replaceChildren(h("p", { class: "callout", text: "No data: run python main.py --web 1 to create web/data.js." }));
} else {
  overview();
  models();
  explain();
  pitLoss();
  strategy();
  validation();
  safetyCar();
  undercut();
  traffic();
  document.getElementById("footer").textContent =
    `Data: OpenF1 (openf1.org). Generated ${DATA.generated} by python main.py --web 1. Model: tuned monotone XGBoost.`;
  setupTheme();
  setupNav();
  renderAll();
  let pending;
  window.addEventListener("resize", () => {
    clearTimeout(pending);
    pending = setTimeout(renderAll, 150);
  });
}
