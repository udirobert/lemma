// Discovery replay engine — plays public/demo/<slug>.json into the
// mission-control layout. Every event is real (see data file `provenance`);
// this file only handles timing + DOM.

interface DemoEvent {
  t: number;
  agent: string;
  kind: string;
  title: string;
  body?: string;
  claim?: string;
  status?: string;
  verdict?: string;
  score?: string;
  big?: string;
  claims?: Array<{ id: string; label: string; kind: string }>;
  chapter: string;
}

interface DemoData {
  provenance: string;
  paper: { id: string; title: string; url: string; tagline: string };
  agents: Array<{ id: string; label: string; sub: string }>;
  chapters: Array<{ id: string; t: number; n: number; title: string }>;
  events: DemoEvent[];
  finale: {
    verdicts: Array<{ claim: string; label: string; verdict: string; note: string; generated?: boolean }>;
    metrics: Array<{ k: string; v: string }>;
  };
}

const PHASES = ["question", "evidence", "hypothesis", "experiment", "result", "updated decision"];

function phaseFor(ev: DemoEvent): string {
  const ch = ev.chapter;
  if (ch === "c1") return "question";
  if (ch === "c2" || ch === "c3") return "evidence";
  if (ch === "c4") return ev.kind === "attempt" || ev.kind === "verdict" ? "experiment" : "hypothesis";
  if (ch === "c5") return "result";
  return "updated decision";
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const AGENT_SHORT: Record<string, string> = {
  director: "director", auditor: "auditor", proposer: "proposer",
  engine: "engine", human: "human", judge: "judge",
};

let DATA: DemoData;
let playing = false;
let clock = 0;            // replay seconds elapsed
let lastTs = 0;
let eventIdx = 0;
let done = false;
let duration = 108;
let counts = { attempts: 0, failed: 0, verdicts: 0 };
let suppressFx = false;   // during scrub catch-up

const KIND_VERB: Record<string, string> = {
  note: "note", cmd: "cmd", claims: "claims", dispatch: "routes work",
  attempt: "audit attempt", strike: "attempt failed", verdict: "verdict",
  feedback: "feedback", incident: "incident", proposal: "proposes",
  promote: "promotes claim", caveat: "caveat", gate: "approval gate",
  judge: "verdict", metric: "metric", summary: "summary",
};

async function boot() {
  DATA = await (await fetch("/demo/double-descent.json")).json();
  duration = Math.max(...DATA.events.map((e) => e.t)) + 6;
  const tm = Math.floor(duration / 60), ts = Math.floor(duration % 60);
  $("d-clock").textContent = `0:00 / ${tm}:${String(ts).padStart(2, "0")}`;
  buildRails();
  buildTicks();
  $("d-play").addEventListener("click", startReplay);
  $("d-pp").addEventListener("click", toggle);
  $("d-track").addEventListener("click", onSeek);
  if (new URLSearchParams(location.search).has("auto")) startReplay();
}

function buildRails() {
  const rail = $("d-agents");
  rail.innerHTML = `<div class="d-loop-head">agents · omnigent</div>` +
    DATA.agents.map((a) =>
      `<div class="d-agent" data-agent="${a.id}">
        <span class="a-name">${a.label}</span>
        <span class="a-sub">${a.sub}</span>
        <span class="a-last" id="al-${a.id}"></span>
      </div>`).join("");

  $("d-loop").innerHTML = `<div class="d-loop-head">discovery loop</div>` +
    PHASES.map((p, i) =>
      `<div class="ph" data-ph="${p}"><span class="ph-n">${i + 1}·</span>${p}</div>`).join("");
}

function buildTicks() {
  $("d-ticks").innerHTML = DATA.events.map((e) =>
    `<div class="d-tick${e.kind === "verdict" || e.kind === "judge" ? " major" : ""}"
      data-t="${e.t}" style="left:${(e.t / duration) * 100}%"></div>`).join("");
}

function startReplay() {
  $("d-intro").classList.add("out");
  $("d-console").hidden = false;
  setTimeout(() => { $("d-intro").style.display = "none"; }, 550);
  playing = true;
  lastTs = performance.now();
  requestAnimationFrame(tick);
}

function toggle() {
  if (done) { restart(); return; }
  playing = !playing;
  lastTs = performance.now();
  if (playing) requestAnimationFrame(tick);
  $("d-pp").textContent = playing ? "❚❚" : "▶";
}

function restart() {
  clock = 0; eventIdx = 0; done = false;
  counts = { attempts: 0, failed: 0, verdicts: 0 };
  $("d-stage").innerHTML = "";
  $("d-ledger").innerHTML = "";
  $("d-judge-chip").hidden = true;
  document.querySelectorAll(".d-agent").forEach((a) => a.classList.remove("active"));
  document.querySelectorAll(".ph").forEach((p) => p.classList.remove("now", "done"));
  document.querySelectorAll(".d-tick").forEach((t) => t.classList.remove("done"));
  renderCounters();
  playing = true;
  lastTs = performance.now();
  requestAnimationFrame(tick);
  $("d-pp").textContent = "❚❚";
}

function onSeek(e: MouseEvent) {
  const r = $("d-track").getBoundingClientRect();
  const target = ((e.clientX - r.left) / r.width) * duration;
  // reset then catch up instantly
  suppressFx = true;
  clock = 0; eventIdx = 0; done = false;
  counts = { attempts: 0, failed: 0, verdicts: 0 };
  $("d-stage").innerHTML = "";
  $("d-ledger").innerHTML = "";
  $("d-judge-chip").hidden = true;
  document.querySelectorAll(".d-tick").forEach((t) => t.classList.remove("done"));
  document.querySelectorAll(".ph").forEach((p) => p.classList.remove("now", "done"));
  while (eventIdx < DATA.events.length && DATA.events[eventIdx].t <= target) {
    fire(DATA.events[eventIdx]);
    eventIdx++;
  }
  clock = target;
  suppressFx = false;
  if (!playing) { toggle(); }
  lastTs = performance.now();
}

function tick(now: number) {
  if (!playing) return;
  const dt = (now - lastTs) / 1000;
  lastTs = now;
  clock += dt;
  while (eventIdx < DATA.events.length && DATA.events[eventIdx].t <= clock) {
    fire(DATA.events[eventIdx]);
    eventIdx++;
  }
  updateChrome();
  if (eventIdx >= DATA.events.length && clock >= duration) {
    finish();
    return;
  }
  requestAnimationFrame(tick);
}

function updateChrome() {
  const pct = Math.min(100, (clock / duration) * 100);
  $("d-track-fill").style.width = pct + "%";
  const mm = Math.floor(clock / 60), ss = Math.floor(clock % 60);
  const tm = Math.floor(duration / 60), ts = Math.floor(duration % 60);
  $("d-clock").textContent = `${mm}:${String(ss).padStart(2, "0")} / ${tm}:${String(ts).padStart(2, "0")}`;
  $("c-wall").textContent = `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
  const ch = [...DATA.chapters].reverse().find((c) => clock >= c.t);
  if (ch) $("d-chapter").textContent = `${ch.n} / ${DATA.chapters.length} · ${ch.title}`;
  document.querySelectorAll<HTMLElement>(".d-tick").forEach((t) => {
    t.classList.toggle("done", parseFloat(t.dataset.t || "0") <= clock);
  });
}

function renderCounters() {
  $("c-attempts").textContent = String(counts.attempts);
  $("c-failed").textContent = String(counts.failed);
  $("c-verdicts").textContent = String(counts.verdicts);
}

function setActiveAgent(id: string, lastText: string) {
  document.querySelectorAll(".d-agent").forEach((a) => a.classList.remove("active"));
  const el = document.querySelector(`.d-agent[data-agent="${id}"]`);
  el?.classList.add("active");
  const last = $(`al-${id}`);
  if (last) last.textContent = lastText;
}

function setPhase(name: string) {
  const idx = PHASES.indexOf(name);
  document.querySelectorAll<HTMLElement>(".ph").forEach((p) => {
    const pi = PHASES.indexOf(p.dataset.ph || "");
    p.classList.toggle("done", pi < idx);
    p.classList.toggle("now", pi === idx);
  });
}

function ledger(ev: DemoEvent) {
  const row = document.createElement("div");
  row.className = `lg ${ev.verdict || ev.kind} ${ev.agent === "human" ? "human" : ""}`;
  const mm = Math.floor(ev.t / 60), ss = Math.floor(ev.t % 60);
  row.innerHTML = `<span class="lt">${mm}:${String(ss).padStart(2, "0")}</span>
    <span class="la">${AGENT_SHORT[ev.agent] || ev.agent}</span>
    <span class="lm">${ev.title}</span>`;
  const lg = $("d-ledger");
  lg.appendChild(row);
  lg.scrollTop = lg.scrollHeight;
}

function card(inner: string, cls = "") {
  const stage = $("d-stage");
  const prev = stage.querySelector(".d-card, .d-finale");
  if (prev) {
    if (suppressFx) prev.remove();
    else { prev.classList.add("leaving"); setTimeout(() => prev.remove(), 300); }
  }
  const el = document.createElement("div");
  el.className = `d-card ${cls}`;
  if (suppressFx) el.style.animation = "none";
  el.innerHTML = inner;
  stage.appendChild(el);
}

function kick(ev: DemoEvent, extra = "") {
  return `<div class="card-kick"><span class="who">${AGENT_SHORT[ev.agent] || ev.agent}</span>${extra}<span class="verb">· ${KIND_VERB[ev.kind] || ev.kind}</span></div>`;
}

function fire(ev: DemoEvent) {
  setActiveAgent(ev.agent, ev.title.slice(0, 40));
  setPhase(phaseFor(ev));
  ledger(ev);

  switch (ev.kind) {
    case "claims":
      card(`${kick(ev)}<h2>${ev.title}</h2>
        <div class="d-claimgrid">${(ev.claims || []).map((c, i) =>
          `<div class="d-chip" style="animation-delay:${i * 0.08}s">
            <div class="ch-id">${c.id} <span class="ch-kind">${c.kind}</span></div>
            <div class="ch-label">${c.label}</div>
          </div>`).join("")}</div>`);
      break;

    case "attempt":
      counts.attempts++; renderCounters();
      card(`${kick(ev, `<span class="mono" style="color:var(--blue)">${ev.claim}</span>`)}
        <h2>${ev.title}</h2><p>${ev.body}</p>`, "");
      break;

    case "strike":
      counts.attempts++; counts.failed++; renderCounters();
      card(`${kick(ev, `<span class="mono" style="color:var(--magenta)">${ev.claim}</span>`)}
        <h2>${ev.title}</h2><p>${ev.body}</p><div class="strike-x">✕</div>`,
        ev.status === "crash" ? "strike-card crash-card" : "strike-card");
      break;

    case "verdict":
      counts.verdicts++; renderCounters();
      card(`<div class="d-stamp ${ev.verdict}">${ev.verdict}</div>
        ${kick(ev, `<span class="mono">${ev.claim}</span>`)}
        <h2>${ev.title}</h2><p>${ev.body}</p>`, "verdict-card");
      break;

    case "feedback":
      card(`${kick(ev)}<h2>${ev.title}</h2><p class="quote">${ev.body}</p>`, "human-card");
      break;

    case "promote":
      card(`${kick(ev)}<h2><span class="gen-tag">generated hypothesis</span> ${ev.title}</h2><p>${ev.body}</p>`, "gen-card");
      break;

    case "proposal":
      card(`${kick(ev)}<h2>${ev.title}</h2><p>${ev.body}</p>`, "gen-card");
      break;

    case "gate":
      card(`${kick(ev)}<h2><span class="gate-lock">🔒</span> ${ev.title}</h2><p>${ev.body}</p>`, "gate-card");
      break;

    case "caveat":
      card(`${kick(ev)}<h2>${ev.title}</h2><p>${ev.body}</p>`, "human-card");
      break;

    case "judge":
      $("d-judge-chip").hidden = false;
      card(`<div class="d-stamp supported">${ev.verdict} ${ev.score}</div>
        ${kick(ev)}<h2>${ev.title}</h2><p>${ev.body}</p>
        <div class="judge-rows">
          <div class="judge-row"><span>structure · claims + report present</span><b>PASS</b></div>
          <div class="judge-row"><span>evidence · 3 audited claims</span><b>PASS</b></div>
          <div class="judge-row"><span>integrity · 0 silently skipped</span><b>PASS</b></div>
          <div class="judge-row"><span>cost disclosure · wall time logged</span><b>PASS</b></div>
          <div class="judge-row"><span>trace · 156 events, failures kept</span><b>PASS</b></div>
        </div>`, "judge-card");
      break;

    case "metric":
      card(`${kick(ev)}<div class="metric-big">${ev.big}</div><h2>${ev.title}</h2><p>${ev.body}</p>`, "metric-card");
      break;

    case "incident":
      card(`${kick(ev)}<h2>${ev.title}</h2><p>${ev.body}</p>`, "crash-card");
      break;

    default:
      card(`${kick(ev)}<h2>${ev.title}</h2><p>${ev.body || ""}</p>`);
  }
}

function finish() {
  done = true;
  playing = false;
  $("d-pp").textContent = "↺";
  $("d-chapter").textContent = "finale · the trail is the deliverable";

  const f = DATA.finale;
  const stage = $("d-stage");
  stage.innerHTML = `<div class="d-finale">
    <h2>the evidence trail — everything reconstructible</h2>
    <div class="fin-verdicts">${f.verdicts.map((v) => `
      <div class="fin-v ${v.verdict}">
        ${v.generated ? '<span class="gen-tag">agent-generated</span>' : ""}
        <div class="fv-id">${v.claim}</div>
        <div class="fv-label">${v.label}</div>
        <div class="fv-verdict">${v.verdict.replace("_", " ")}</div>
        <div class="fv-note">${v.note}</div>
      </div>`).join("")}
    </div>
    <div class="fin-metrics">${f.metrics.map((m) => `
      <div class="fin-m"><div class="fm-k">${m.k}</div><div class="fm-v">${m.v}</div></div>`).join("")}
    </div>
    <div class="fin-links">
      <a href="/papers/double-descent/">full audit page ↗</a>
      <a href="/room/?paper=double-descent">audit room ↗</a>
      <a href="https://arxiv.org/abs/1912.07242" target="_blank" rel="noopener">the paper ↗</a>
      <a href="https://github.com/udirobert/lemma" target="_blank" rel="noopener">repository ↗</a>
    </div>
  </div>`;
}

boot();
