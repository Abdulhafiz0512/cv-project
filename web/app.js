"use strict";

const CLASS_INFO = {
  accident: ["#e5484d", "Two road users' footprints meet, then both stop abruptly and stay put for 3 s, outside the signal queue."],
  near_miss: ["#f76b15", "A vehicle brakes hard (over 3 box heights/s²) with a road user close ahead, and they never touch."],
  red_light: ["#ff8fa3", "A vehicle crosses the inbound stop line while another vehicle is waiting at it before and after the crossing."],
  wrong_way: ["#d864d8", "Moving against the one-way direction of the divided carriageway for 1.5 s and 2.5 box heights."],
  illegal_u_turn: ["#a78bfa", "The heading turns by 150° or more on the carriageway, with no identity jump in the track."],
  stopped_vehicle: ["#f5d90a", "Stationary 10 s or more on a road link: not in the signal queue, the junction, a zebra approach, parking or the bus bay, and not queued behind another car."],
  jaywalking: ["#4cc38a", "A person on foot (slower than an e-scooter) at least half a body height inside the carriageway and away from every zebra for 1.2 s."],
  failure_to_yield: ["#3ecfe0", "A vehicle, or a ridden two-wheeler, drives through a zebra while a pedestrian on the same zebra is within three vehicle heights."],
  illegal_turn: ["#8da4ef", null],
  solid_line_crossing: ["#f0f0f0", "The ground point crosses the solid line along the median."],
  stop_line: ["#ffb224", "A vehicle stands still for 3 s past the stop line, between it and the far edge of the zebra."],
  congestion: ["#b08a5a", "Six or more vehicles in a one-way carriageway, 80% of them crawling, for longer than a signal cycle."],
  road_obstacle: ["#9ba1a6", "An animal on the carriageway for 2 s."],
  fire_smoke: ["#ff5a1f", null],
};
const NOT_EMITTED = "Not predicted: it cannot be told reliably from this view, and a wrong prediction of a class that never occurs costs a whole class in the score.";

const $ = (sel) => document.querySelector(sel);
const svgNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}
function fmt(sec) {
  const m = Math.floor(sec / 60), s = sec - 60 * m;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
async function getJSON(url) {
  try {
    const r = await fetch(url, { cache: "no-cache" });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}
function niceStep(span, target = 8) {
  const raw = span / target, p = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
}

/* ---------------------------------------------------------------- demo */
function renderDemo(url) {
  const slot = $("#demo-slot");
  if (url) {
    const frame = document.createElement("iframe");
    frame.src = url;
    frame.title = "Live demo";
    frame.loading = "lazy";
    frame.allow = "clipboard-write";
    slot.appendChild(frame);
    const p = document.createElement("p");
    p.innerHTML = `If the embedded demo does not load, <a href="${url}">open it in its own tab</a>.`;
    slot.appendChild(p);
  } else {
    slot.innerHTML = '<p class="empty">The demo address is not configured yet. Run <code>python demo/app.py</code> from the repository to use it locally.</p>';
  }
}

/* ---------------------------------------------------------------- rules table */
function renderRules(classes) {
  const body = $("#rules-body");
  for (const c of classes) {
    const [color, rule] = CLASS_INFO[c] || ["#999", null];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><span class="swatch" style="background:${color}"></span>${c}</td>` +
      `<td class="${rule ? "" : "off"}">${rule || NOT_EMITTED}</td>`;
    body.appendChild(tr);
  }
}

/* ---------------------------------------------------------------- tabs */
function tabs(container, items, onSelect) {
  container.innerHTML = "";
  const buttons = items.map((label, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.role = "tab";
    b.textContent = label;
    b.addEventListener("click", () => select(i));
    container.appendChild(b);
    return b;
  });
  function select(i) {
    buttons.forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
    onSelect(i);
  }
  if (items.length) select(0);
}

/* ---------------------------------------------------------------- results */
let playheads = [];
function renderResults(site) {
  const videos = site.videos || [];
  const player = $("#player");
  if (!videos.length) {
    $("#player-empty").hidden = false;
    $("#player-empty").textContent = "Results appear here after tools/build_site.py has run on the sample videos.";
    player.hidden = true;
    return;
  }
  player.addEventListener("timeupdate", () => playheads.forEach((fn) => fn(player.currentTime)));
  tabs($("#video-tabs"), videos.map((v) => v.stem), (i) => {
    const v = videos[i];
    const duration = v.meta.duration_s || (v.risk.length ? v.risk[v.risk.length - 1][0] : 1);
    if (v.annotated) {
      player.hidden = false;
      $("#player-empty").hidden = true;
      player.src = v.annotated;
    } else {
      player.hidden = true;
      $("#player-empty").hidden = false;
      $("#player-empty").textContent = "No annotated render for this video.";
    }
    const seek = (t) => { if (!player.hidden) { player.currentTime = t; player.play().catch(() => {}); } };
    playheads = [drawTimeline($("#timeline"), v.events, duration, site.classes, seek),
                 drawRisk($("#riskline"), v.risk, duration)];
    fillEvents(v.events, seek);
  });
}

function drawTimeline(svg, events, duration, classes, seek) {
  svg.innerHTML = "";
  const labels = classes.filter((c) => events.some((e) => e[2] === c));
  const W = 1000, left = 150, right = 12, laneH = 26, top = 8;
  const H = top + Math.max(1, labels.length) * laneH + 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (t) => left + (W - left - right) * t / Math.max(duration, 1e-6);
  if (!labels.length) {
    el("text", { x: left, y: top + 18 }, svg).textContent = "No events detected in this video.";
  }
  labels.forEach((lab, i) => {
    const y = top + i * laneH;
    el("text", { x: 0, y: y + 17 }, svg).textContent = lab;
    el("line", { x1: left, x2: W - right, y1: y + laneH - 2, y2: y + laneH - 2, class: "lane-line" }, svg);
    for (const [s, e, l] of events) {
      if (l !== lab) continue;
      const r = el("rect", { x: x(s), y: y + 4, width: Math.max(3, x(e) - x(s)), height: laneH - 10, rx: 2,
        fill: (CLASS_INFO[l] || ["#999"])[0], class: "seg", tabindex: 0, role: "button",
        "aria-label": `${l} from ${fmt(s)} to ${fmt(e)}` }, svg);
      el("title", {}, r).textContent = `${l}: ${fmt(s)} – ${fmt(e)}`;
      r.addEventListener("click", () => seek(s));
      r.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); seek(s); } });
    }
  });
  const axisY = top + Math.max(1, labels.length) * laneH + 16;
  const step = niceStep(duration);
  for (let t = 0; t <= duration + 1e-6; t += step) {
    el("text", { x: x(t), y: axisY, "text-anchor": "middle" }, svg).textContent = fmt(t).replace(/\.0$/, "");
  }
  const head = el("line", { x1: x(0), x2: x(0), y1: 0, y2: axisY - 12, class: "playhead" }, svg);
  return (t) => { head.setAttribute("x1", x(t)); head.setAttribute("x2", x(t)); };
}

function drawRisk(svg, risk, duration) {
  svg.innerHTML = "";
  const W = 1000, H = 90, left = 150, right = 12, top = 8, bottom = 18;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (t) => left + (W - left - right) * t / Math.max(duration, 1e-6);
  const y = (v) => top + (H - top - bottom) * (1 - v);
  el("text", { x: 0, y: y(0.5) + 4 }, svg).textContent = "accident risk";
  el("line", { x1: left, x2: W - right, y1: y(0), y2: y(0), class: "axis" }, svg);
  el("line", { x1: left, x2: W - right, y1: y(0.5), y2: y(0.5), class: "threshold" }, svg);
  el("text", { x: W - right, y: y(0.5) - 4, "text-anchor": "end" }, svg).textContent = "alarm at 0.5";
  if (risk.length) {
    el("polyline", { class: "curve", points: risk.map(([t, v]) => `${x(t).toFixed(1)},${y(v).toFixed(1)}`).join(" ") }, svg);
  }
  const head = el("line", { x1: x(0), x2: x(0), y1: top, y2: y(0), class: "playhead" }, svg);
  return (t) => { head.setAttribute("x1", x(t)); head.setAttribute("x2", x(t)); };
}

function fillEvents(events, seek) {
  const body = $("#events-table tbody");
  body.innerHTML = "";
  if (!events.length) {
    body.innerHTML = '<tr><td colspan="4" class="empty">No events in this video.</td></tr>';
    return;
  }
  for (const [s, e, l] of [...events].sort((a, b) => a[0] - b[0])) {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.innerHTML = `<td><span class="swatch" style="background:${(CLASS_INFO[l] || ["#999"])[0]}"></span>${l}</td>` +
      `<td>${fmt(s)}</td><td>${fmt(e)}</td><td>${(e - s).toFixed(1)} s</td>`;
    tr.addEventListener("click", () => seek(s));
    tr.addEventListener("keydown", (ev) => { if (ev.key === "Enter") seek(s); });
    body.appendChild(tr);
  }
}

/* ---------------------------------------------------------------- EDA */
function renderEDA(site) {
  const videos = site.videos || [];
  if (!videos.length) return;
  tabs($("#eda-tabs"), videos.map((v) => v.stem), async (i) => {
    const v = videos[i];
    const m = v.meta;
    const rows = [
      ["Resolution", `${m.width}×${m.height}`], ["Frame rate", `${m.fps} fps`], ["Duration", fmt(m.duration_s || 0)],
      ["Codec", `${m.codec} ${m.profile || ""}`], ["Pixel format", m.pix_fmt], ["Bitrate", `${m.bitrate_mbps} Mbit/s`],
      ["Vehicle tracks", v.tracks.vehicle], ["Two-wheeler tracks", v.tracks.two_wheeler], ["Pedestrian tracks", v.tracks.person],
    ];
    $("#eda-meta").innerHTML = rows.map(([k, val]) => `<div><dt>${k}</dt><dd>${val}</dd></div>`).join("");
    $("#eda-background").src = v.eda.background;
    $("#eda-flow").src = v.eda.flow_field;
    $("#eda-traj").src = v.eda.trajectories;
    $("#eda-heat-veh").src = v.eda.heat_vehicles;
    $("#eda-heat-ped").src = v.eda.heat_pedestrians;
    const st = await getJSON(v.stats);
    if (!st) return;
    const secs = st.counts_per_second.vehicle.map((_, j) => j);
    lineChart($("#chart-counts"), secs, [
      { name: "vehicles", ys: st.counts_per_second.vehicle, color: "var(--sign-blue)" },
      { name: "pedestrians", ys: st.counts_per_second.person, color: "#2f9e6e" },
      { name: "two-wheelers", ys: st.counts_per_second.two_wheeler, color: "var(--amber)" },
    ]);
    const e = st.speed_hist_bhs.edges;
    barChart($("#chart-speed"), e.slice(0, -1).map((a, j) => (a + e[j + 1]) / 2), st.speed_hist_bhs.counts);
    lineChart($("#chart-light"), st.lighting.map((p) => p[0]), [{ name: "brightness", ys: st.lighting.map((p) => p[1]), color: "var(--muted)" }]);
  });
}

function frame(svg, xs, ymax) {
  svg.innerHTML = "";
  const W = 400, H = 200, l = 34, r = 8, t = 10, b = 24;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const xmin = xs[0] ?? 0, xmax = xs[xs.length - 1] ?? 1;
  const x = (v) => l + (W - l - r) * (v - xmin) / Math.max(xmax - xmin, 1e-6);
  const y = (v) => t + (H - t - b) * (1 - v / Math.max(ymax, 1e-6));
  const ys = niceStep(ymax, 4);
  for (let v = 0; v <= ymax + 1e-6; v += ys) {
    el("line", { x1: l, x2: W - r, y1: y(v), y2: y(v), class: "grid" }, svg);
    el("text", { x: l - 4, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = +v.toFixed(2);
  }
  const xsStep = niceStep(xmax - xmin, 5);
  for (let v = Math.ceil(xmin / xsStep) * xsStep; v <= xmax + 1e-6; v += xsStep) {
    el("text", { x: x(v), y: H - 6, "text-anchor": "middle" }, svg).textContent = +v.toFixed(1);
  }
  return { x, y, W, r, t };
}
function lineChart(svg, xs, series) {
  const ymax = Math.max(1, ...series.flatMap((s) => s.ys));
  const f = frame(svg, xs, ymax);
  series.forEach((s, i) => {
    el("polyline", { fill: "none", stroke: s.color, "stroke-width": 1.6,
      points: s.ys.map((v, j) => `${f.x(xs[j]).toFixed(1)},${f.y(v).toFixed(1)}`).join(" ") }, svg);
    const lx = f.W - f.r - 4, ly = f.t + 12 + 14 * i;
    el("text", { x: lx, y: ly, "text-anchor": "end", fill: s.color, style: `fill:${s.color}` }, svg).textContent = s.name;
  });
}
function barChart(svg, xs, ys) {
  const f = frame(svg, xs, Math.max(1, ...ys));
  const w = xs.length > 1 ? (f.x(xs[1]) - f.x(xs[0])) * 0.8 : 10;
  xs.forEach((v, j) => el("rect", { x: f.x(v) - w / 2, y: f.y(ys[j]), width: w, height: f.y(0) - f.y(ys[j]), fill: "var(--sign-blue)" }, svg));
}

/* ---------------------------------------------------------------- team and links */
function renderTeam(team) {
  const list = $("#team-list");
  if (!team || !team.members) return;
  for (const m of team.members) {
    const d = document.createElement("article");
    d.className = "person";
    const links = [["GitHub", m.github], ["LinkedIn", m.linkedin], ["Portfolio", m.portfolio]].filter(([, u]) => u);
    d.innerHTML = `<h3>${m.name}</h3><p class="role">${m.role}</p>` +
      `<ul>${(m.contributions || []).map((c) => `<li>${c}</li>`).join("")}</ul>` +
      (m.projects && m.projects.length ? `<p>Proud of: ${m.projects.map((p) => p.url ? `<a href="${p.url}">${p.name}</a>` : p.name).join(", ")}</p>` : "") +
      `<p class="links">${links.map(([k, u]) => `<a href="${u}">${k}</a>`).join("")}</p>`;
    list.appendChild(d);
  }
}
function renderLinks(site) {
  const repo = site.repo_url;
  const items = [
    repo ? `<a href="${repo}">Source code repository</a>` : "Source code repository: add the URL in tools/build_site.py --repo-url",
    repo ? `<a href="${repo}/tree/main/weights">Model weights</a> (YOLO26-S, 20 MB, committed to the repository)` : "Model weights: weights/yolo26s.pt in the repository",
    `<a href="data/predictions_samples.json">predictions_samples.json</a>, our output on the sample videos`,
    site.demo_url ? `<a href="${site.demo_url}">Live demo</a>` : "Live demo: see above",
  ];
  $("#links-list").innerHTML = items.map((i) => `<li>${i}</li>`).join("");
}

function renderAblations(ab) {
  if (!ab) return;
  $("#ablation-detectors tbody").innerHTML = ab.detectors.map((r) =>
    `<tr><td>${r.detector}</td><td>${r.imgsz} px</td><td>${r.ms_per_frame}</td>` +
    `<td>${r.per_frame.vehicle + r.per_frame.two_wheeler}</td><td>${r.per_frame.person}</td>` +
    `<td>${r.tracks.vehicle + r.tracks.two_wheeler + r.tracks.person}</td><td>${r.events}</td></tr>`).join("");
  $("#ablation-decode tbody").innerHTML = ab.decode.map((r) =>
    `<tr><td>${r.reader}</td><td>${r.frames_out}</td><td>${r.realtime_factor}</td></tr>`).join("");
}

(async function main() {
  const site = (await getJSON("data/site.json")) || { classes: Object.keys(CLASS_INFO), videos: [] };
  const team = await getJSON("data/team.json");
  renderDemo(site.demo_url);
  renderRules(site.classes || Object.keys(CLASS_INFO));
  renderResults(site);
  renderEDA(site);
  renderTeam(team);
  renderLinks(site);
  renderAblations(await getJSON("data/ablations.json"));
})();
