"use strict";

const TYPE_LABEL = { real_time: "Real-Time", proactive: "Proactive", multi_response: "Multi-Response" };
const TYPE_COLOR = { real_time: "var(--rt)", proactive: "var(--pro)", multi_response: "var(--multi)" };

const $ = (id) => document.getElementById(id);
let SUMMARY = null;
let ACTIVE = null;

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status + " " + url);
  return r.json();
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function badge(type) {
  const b = el("span", "badge " + type, TYPE_LABEL[type] || type);
  return b;
}

async function loadSummary() {
  SUMMARY = await getJSON("/api/summary");
  $("workdir").textContent = SUMMARY.work_dir;
  $("dataset-hint").textContent = SUMMARY.has_jsonl
    ? `${SUMMARY.n_videos_with_frames}/${SUMMARY.n_videos} videos have frames on disk`
    : "no training_instances.jsonl found";

  const sup = SUMMARY.supervision || { silent_fraction: 0 };
  const cards = $("cards");
  cards.innerHTML = "";
  cards.appendChild(metric("Training instances", String(SUMMARY.n_instances)));
  cards.appendChild(metric("Source videos", String(SUMMARY.n_videos)));
  cards.appendChild(metric("Silent supervision", (sup.silent_fraction * 100).toFixed(1) + "%"));

  const tc = el("div", "card");
  tc.appendChild(withLabel("By QA type"));
  const bars = el("div", "typebars");
  const order = ["real_time", "proactive", "multi_response"];
  order.forEach((t) => {
    const n = (SUMMARY.by_type || {})[t] || 0;
    const row = el("div", "typebar");
    const dot = el("span", "dot"); dot.style.background = TYPE_COLOR[t];
    row.appendChild(dot);
    row.appendChild(el("span", null, TYPE_LABEL[t]));
    row.appendChild(el("span", "n", String(n)));
    bars.appendChild(row);
  });
  tc.appendChild(bars);
  cards.appendChild(tc);

  const ft = $("filter-type");
  ft.innerHTML = '<option value="">All QA types</option>';
  (SUMMARY.types || []).forEach((t) => {
    const o = el("option", null, TYPE_LABEL[t] || t); o.value = t; ft.appendChild(o);
  });
  const fv = $("filter-video");
  fv.innerHTML = '<option value="">All videos</option>';
  (SUMMARY.videos || []).forEach((v) => {
    const o = el("option", null, v); o.value = v; fv.appendChild(o);
  });
}

function metric(label, value) {
  const c = el("div", "card");
  c.appendChild(withLabel(label));
  c.appendChild(el("div", "value", value));
  return c;
}
function withLabel(t) { return el("div", "label", t); }

async function loadList() {
  const params = new URLSearchParams();
  if ($("filter-type").value) params.set("type", $("filter-type").value);
  if ($("filter-video").value) params.set("video", $("filter-video").value);
  if ($("search").value.trim()) params.set("q", $("search").value.trim());
  const items = await getJSON("/api/instances?" + params.toString());
  $("list-count").textContent = `${items.length} instance${items.length === 1 ? "" : "s"}`;
  const pane = $("listpane");
  pane.innerHTML = "";
  if (!items.length) { pane.appendChild(el("div", "empty", "No instances match.")); return; }
  items.forEach((it) => {
    const li = el("button", "li");
    li.dataset.id = it.instance_id;
    const head = el("div", "li-head");
    head.appendChild(badge(it.source_qa_type));
    if (it.has_frames) head.appendChild(el("span", "chip", "frames"));
    li.appendChild(head);
    li.appendChild(el("div", "li-q", it.first_question || "(silent-anchored instance)"));
    const meta = el("div", "li-meta");
    meta.appendChild(el("span", null, it.video_id));
    meta.appendChild(el("span", null, `${it.n_turns} turns`));
    meta.appendChild(el("span", null, `${it.n_chunks} chunks`));
    li.appendChild(meta);
    li.onclick = () => selectInstance(it.instance_id, li);
    pane.appendChild(li);
  });
}

async function selectInstance(id, liEl) {
  document.querySelectorAll(".li.active").forEach((e) => e.classList.remove("active"));
  if (liEl) liEl.classList.add("active");
  ACTIVE = id;
  const d = await getJSON("/api/instance/" + encodeURIComponent(id));
  renderDetail(d);
}

function renderDetail(d) {
  const pane = $("detailpane");
  pane.innerHTML = "";
  const silentWeight = 1 / Math.max(1, d.n_silent_supervised);

  const head = el("div", "detail-head");
  head.appendChild(badge(d.source_qa_type));
  head.appendChild(el("span", "id", d.instance_id));
  pane.appendChild(head);

  const sub = el("div", "detail-sub");
  sub.appendChild(el("span", null, "video: " + d.video_id));
  sub.appendChild(el("span", null, `target chunk: #${d.target_chunk_index}`));
  sub.appendChild(el("span", null, `N_silent: ${d.n_silent_supervised}  (silent weight ${silentWeight.toFixed(3)})`));
  if (d.quality_passed != null) sub.appendChild(el("span", null, "quality: " + (d.quality_passed ? "passed" : "filtered")));
  pane.appendChild(sub);

  pane.appendChild(el("div", "section-label", "Supervision mask over " + d.chunks.length + " one-second chunks"));
  const strip = el("div", "strip");
  d.chunks.forEach((c) => {
    let cls = "cell ctx";
    if (c.is_target) cls = "cell tgt";
    else if (c.supervised && c.is_silent) cls = "cell sil";
    const cell = el("div", cls, String(c.t_s));
    cell.title = `t=${c.t_s}s · ${c.is_target ? "target" : (c.supervised ? "silent, supervised" : (c.text_only ? "history (text only)" : "spoken, context only"))} · weight ${c.weight}`;
    strip.appendChild(cell);
  });
  pane.appendChild(strip);
  const legend = el("div", "legend");
  legend.innerHTML =
    '<span><i style="background:var(--accent-bg)"></i>silent, supervised (w ' + silentWeight.toFixed(3) + ')</span>' +
    '<span><i style="background:var(--target)"></i>target (w 1.00)</span>' +
    '<span><i style="background:var(--surface-2);border:1px dashed var(--border-2)"></i>spoken / history, context only (w 0)</span>';
  pane.appendChild(legend);

  pane.appendChild(el("div", "section-label", "Conversation turns (video frame + text at each timestamp)"));
  const turns = el("div", "turns");
  const speaking = d.chunks.filter((c) => c.user_text || !c.is_silent);
  speaking.forEach((c) => {
    const row = el("div", "turn" + (c.is_target ? " tgt" : ""));
    // thumbnail
    if (!c.text_only && d.has_frames) {
      const img = el("img", "thumb");
      img.decoding = "async";
      img.alt = `frame at ${c.t_s}s`;
      img.src = `/api/frame?video=${encodeURIComponent(d.video_id)}&t=${c.t_s}`;
      img.onerror = () => { img.replaceWith(placeholder(c.text_only ? "history" : "no frame")); };
      row.appendChild(img);
    } else {
      row.appendChild(placeholder(c.text_only ? "history" : "no frame"));
    }
    const body = el("div", "tbody");
    const th = el("div", "thead");
    th.appendChild(el("span", "ttime", c.t_s + "s"));
    if (c.qa_type) th.appendChild(badge(c.qa_type));
    if (c.is_acknowledgment) th.appendChild(el("span", "chip", "ack"));
    if (c.text_only) th.appendChild(el("span", "chip", "history"));
    const w = el("span", "wt" + (c.weight > 0 ? " on" : ""), "w " + c.weight);
    th.appendChild(w);
    body.appendChild(th);
    if (c.user_text) {
      const u = el("div", "u"); u.innerHTML = "<b>User</b> "; u.appendChild(document.createTextNode(c.user_text));
      body.appendChild(u);
    }
    if (c.assistant_text) {
      const a = el("div", "a"); a.innerHTML = "<b>Assistant" + (c.is_acknowledgment ? " (ack)" : "") + "</b> ";
      a.appendChild(document.createTextNode(c.assistant_text));
      body.appendChild(a);
    }
    row.appendChild(body);
    turns.appendChild(row);
  });
  pane.appendChild(turns);
}

function placeholder(text) {
  const p = el("div", "thumb ph", text);
  return p;
}

function chart(title, sub, rows, colorFn) {
  const c = el("div", "chart");
  c.appendChild(el("h3", null, title));
  if (sub) c.appendChild(el("div", "csub", sub));
  const bars = el("div", "bars");
  const max = Math.max(1, ...rows.map((r) => r.count));
  rows.forEach((r) => {
    const row = el("div", "bar-row");
    row.appendChild(el("div", "bl", r.label));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = Math.round((r.count / max) * 100) + "%";
    if (colorFn) fill.style.background = colorFn(r);
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("div", "bv", String(r.count)));
    bars.appendChild(row);
  });
  c.appendChild(bars);
  return c;
}

async function renderOverview() {
  const a = await getJSON("/api/analytics");
  const wrap = $("charts");
  wrap.innerHTML = "";
  if (!a.n_instances) { wrap.appendChild(el("div", "empty", "No instances to analyze.")); return; }

  const typeRows = ["real_time", "proactive", "multi_response"]
    .map((t) => ({ label: TYPE_LABEL[t], count: (a.by_type || {})[t] || 0, type: t }))
    .filter((r) => r.count > 0);
  wrap.appendChild(chart("Instances by QA type", `${a.n_instances} total`, typeRows, (r) => TYPE_COLOR[r.type]));

  const videoRows = Object.entries(a.per_video || {})
    .map(([k, v]) => ({ label: k, count: v })).sort((x, y) => y.count - x.count);
  wrap.appendChild(chart("Instances per source video", `${videoRows.length} videos`, videoRows));

  const sup = a.supervision || { silent_fraction: 0 };
  wrap.appendChild(chart("Silent-supervision fraction",
    `dataset-wide ${(sup.silent_fraction * 100).toFixed(1)}% silent — per-instance distribution`,
    a.silent_fraction_hist || []));

  wrap.appendChild(chart("Context size (chunks per instance)",
    "after dual sliding-window truncation", a.chunks_hist || []));
  wrap.appendChild(chart("Assistant turns per instance", "non-silent turns", a.turns_hist || []));
  wrap.appendChild(chart("Video-window span (seconds)", "retained N-second window", a.span_hist || []));

  const q = a.quality || {};
  const qRows = [
    { label: "passed", count: q.passed || 0 },
    { label: "filtered", count: q.filtered || 0 },
    { label: "unknown", count: q.unknown || 0 },
  ].filter((r) => r.count > 0);
  if (qRows.length) wrap.appendChild(chart("Stage-5 quality outcome", "", qRows));
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function showTab(which) {
  ["browse", "overview", "run"].forEach((t) => {
    $("view-" + t).hidden = t !== which;
    $("tab-" + t).classList.toggle("active", t === which);
    $("tab-" + t).setAttribute("aria-selected", String(t === which));
  });
  if (which === "overview") renderOverview();
  if (which === "run") { populateConfigs(); pollJob(); }
}

let configsLoaded = false;
async function populateConfigs(force, selectName) {
  if (configsLoaded && !force) return;
  const cfgs = await getJSON("/api/configs");
  const sel = $("run-config");
  const prev = selectName || sel.value;
  sel.innerHTML = "";
  cfgs.forEach((c) => {
    const o = el("option", null, c.name + (c.desc ? ` — ${c.desc}` : ""));
    o.value = c.name;
    sel.appendChild(o);
  });
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  configsLoaded = true;
}

// ---- VLM config editor -----------------------------------------------------
const CFG_TEMPLATES = {
  ollama: `# Local VLM via Ollama / any OpenAI-compatible server
[agents.default]
provider            = "openai_compatible"
model               = "qwen2.5vl:7b"
base_url            = "http://localhost:11434/v1"
max_frames_per_call = 4
frame_max_side      = 512
max_tokens          = 1024
request_timeout     = 900.0
`,
  bailian: `# Aliyun Bailian (DashScope) Qwen-VL — key comes from the env var, never hard-coded
[agents.default]
provider    = "dashscope"
model       = "qwen-vl-max"
api_key_env = "DASHSCOPE_API_KEY"
max_frames_per_call = 16
max_tokens          = 2048
`,
  routed: `# Multi-agent: strong cloud model generates, cheap local model verifies.
# Roles: scene / generate / verify / refine / quality; unlisted ones use default.
[roles]
generate = "strong"
verify   = "cheap"

[agents.default]
provider = "openai_compatible"
model    = "qwen2.5vl:7b"
base_url = "http://localhost:11434/v1"

[agents.strong]
provider    = "dashscope"
model       = "qwen-vl-max"
api_key_env = "DASHSCOPE_API_KEY"

[agents.cheap]
provider = "openai_compatible"
model    = "qwen2.5vl:3b"
base_url = "http://localhost:11434/v1"
`,
  mock: `# Offline deterministic mock — no model needed; great for testing the pipeline
[agents.default]
provider  = "mock"
seed      = 0
pass_rate = 0.85
`,
};

function cfgMsg(text, ok) {
  const m = $("cfg-msg");
  m.textContent = text;
  m.className = ok == null ? "hint" : ok ? "hint ok" : "hint err";
}

async function openConfigEditor(mode) {
  $("cfg-editor").hidden = false;
  cfgMsg("");
  if (mode === "edit") {
    const name = $("run-config").value;
    if (name === "mock") {
      cfgMsg("'mock' is built-in — creating a copy you can save under a new name.", null);
      $("cfg-name").value = "my-mock.toml";
      $("cfg-text").value = CFG_TEMPLATES.mock;
      return;
    }
    try {
      const d = await getJSON("/api/config?name=" + encodeURIComponent(name));
      $("cfg-name").value = d.name;
      $("cfg-text").value = d.content;
    } catch (e) {
      cfgMsg("could not load config: " + e.message, false);
    }
  } else {
    $("cfg-name").value = "my-vlm.toml";
    $("cfg-text").value = CFG_TEMPLATES[$("cfg-template").value] || CFG_TEMPLATES.ollama;
  }
}

async function validateConfig() {
  const r = await fetch("/api/config/validate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: $("cfg-name").value, content: $("cfg-text").value }),
  });
  const d = await r.json();
  if (d.ok) cfgMsg("Config is valid.", true);
  else cfgMsg("Problems:\n- " + (d.problems || ["unknown error"]).join("\n- "), false);
  return !!d.ok;
}

async function saveConfig() {
  const name = $("cfg-name").value.trim();
  const r = await fetch("/api/config/save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, content: $("cfg-text").value }),
  });
  const d = await r.json();
  if (!r.ok) {
    cfgMsg((d.error || "save failed") + (d.problems ? ":\n- " + d.problems.join("\n- ") : ""), false);
    return;
  }
  await populateConfigs(true, d.saved);
  cfgMsg(`Saved configs/${d.saved} — selected for the next run.`, true);
}

let jobTimer = null;
async function pollJob() {
  let job;
  try { job = await getJSON("/api/job"); } catch (e) { return; }
  if (job.state && job.state !== "idle") renderJob(job);
  if (job.state === "running" || job.state === "starting") {
    jobTimer = setTimeout(pollJob, 1300);
  } else {
    jobTimer = null;
  }
}

function renderJob(job) {
  const p = $("run-progress");
  p.hidden = false;
  const st = job.state || "idle";
  const frac = job.video_n ? Math.min(1, job.video_i / job.video_n) : (st === "done" ? 1 : 0);
  const s = job.stats || {};
  const stat = (k, key) => `<div class="s"><div class="k">${k}</div><div class="v">${s[key] != null ? s[key] : 0}</div></div>`;
  p.innerHTML =
    `<div class="prog-head"><span class="state-badge ${st}">${st}</span>` +
    `<span class="prog-stage">${esc(job.stage)}</span>` +
    `<span class="hint" style="margin-left:auto">${job.elapsed_s || 0}s</span></div>` +
    `<div class="prog-bar"><div style="width:${Math.round(frac * 100)}%"></div></div>` +
    `<div class="hint">video ${job.video_i}/${job.video_n} · config ${esc(job.config)} · → ${esc(job.work_dir)}</div>` +
    `<div class="prog-stats">` +
      stat("scenes", "n_scenes") + stat("RT verified", "n_rt_verified") +
      stat("proactive", "n_proactive_verified") + stat("multi", "n_multi_verified") +
      stat("unrolled", "n_instances_unrolled") + stat("passed", "n_instances_passed_quality") +
    `</div>` +
    (st === "done" ? `<div class="frow actions"><button class="btn primary" id="load-results">Load ${job.n_instances} results into Browse</button></div>` : "") +
    (job.error ? `<div class="hint" style="color:var(--pro)">${esc(job.error)}</div>` : "") +
    `<div class="section-label">log</div><div class="prog-log" id="prog-log"></div>`;
  const lg = $("prog-log");
  lg.textContent = (job.log || []).join("\n");
  lg.scrollTop = lg.scrollHeight;
  if (st === "done") $("load-results").onclick = () => loadResults(job.work_dir);
  const running = st === "starting" || st === "running";
  $("run-start").disabled = running;
  $("run-stop").hidden = !running;
}

async function startRun() {
  const body = {
    src: $("run-src").value.trim(),
    work_dir: $("run-work").value.trim(),
    config: $("run-config").value,
    params: {
      fps: +$("run-fps").value, video_window_n: +$("run-n").value,
      qa_window_m: +$("run-m").value, seed: +$("run-seed").value,
    },
  };
  const r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) { $("run-note").textContent = data.error || "failed to start"; $("run-note").style.color = "var(--pro)"; return; }
  $("run-note").textContent = "Runs Stage 1→5 into the work-dir, then load results.";
  $("run-note").style.color = "";
  if (jobTimer) clearTimeout(jobTimer);
  pollJob();
}

async function stopRun() { try { await fetch("/api/stop", { method: "POST" }); } catch (e) {} }

async function loadResults(workDir) {
  const r = await fetch("/api/load", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ work_dir: workDir }) });
  const data = await r.json();
  if (!r.ok) { $("run-note").textContent = data.error; return; }
  await loadSummary(); await loadList();
  showTab("browse");
}

let searchTimer = null;
function bind() {
  $("filter-type").onchange = loadList;
  $("filter-video").onchange = loadList;
  $("search").oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadList, 200); };
  $("tab-browse").onclick = () => showTab("browse");
  $("tab-overview").onclick = () => showTab("overview");
  $("tab-run").onclick = () => showTab("run");
  $("run-start").onclick = startRun;
  $("run-stop").onclick = stopRun;
  $("cfg-edit").onclick = () => openConfigEditor("edit");
  $("cfg-new").onclick = () => openConfigEditor("new");
  $("cfg-template").onchange = () => { $("cfg-text").value = CFG_TEMPLATES[$("cfg-template").value]; cfgMsg(""); };
  $("cfg-validate").onclick = validateConfig;
  $("cfg-save").onclick = saveConfig;
  $("cfg-close").onclick = () => { $("cfg-editor").hidden = true; };
  $("reload").onclick = async () => {
    await fetch("/api/reload");
    await loadSummary();
    await loadList();
    if (!$("view-overview").hidden) renderOverview();
    $("detailpane").innerHTML = '<div class="empty">Reloaded. Select an instance.</div>';
  };
}

(async function init() {
  bind();
  await loadSummary();
  await loadList();
})();
