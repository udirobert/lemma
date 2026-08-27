/* Trace player engine: replays /traces/*.json line by line with a clock,
   scrubber, run switcher, and one-time autoplay when scrolled into view. */

interface TraceLine {
  t: number;
  c: string;
  m: string;
  k?: number;
}

interface TraceData {
  key: string;
  title: string;
  source: string;
  judge: string;
  final: { supported: number; falsified: number; inconclusive: number };
  n_events: number;
  n_llm: number;
  n_runs: number;
  wall_min: number;
  lines: TraceLine[];
}

const body = document.getElementById("player-body")!;
const scrub = document.getElementById("player-scrub") as HTMLInputElement;
const ppBtn = document.getElementById("pp-btn")!;
const ppIcon = document.getElementById("pp-icon")!;
const clock = document.getElementById("player-clock")!;
const counter = document.getElementById("player-count")!;
const termTitle = document.getElementById("term-title")!;
const metaSrc = document.getElementById("meta-src")!;
const metaStats = document.getElementById("meta-stats")!;
const finalPanel = document.getElementById("player-final")!;
const finalBadge = document.getElementById("final-badge")!;
const finalText = document.getElementById("final-text")!;

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

function track(name: string, props: Record<string, string> = {}) {
  (
    window as Window & {
      plausible?: (
        n: string,
        o?: { props?: Record<string, string> }
      ) => void;
    }
  ).plausible?.(name, { props });
}

let data: TraceData | null = null;
const cache = new Map<string, TraceData>();
let playing = false;
let playhead = 0; // number of lines shown
let raf = 0;
let lastTs = 0;
let lineStart: number[] = []; // cumulative start time per line
let totalDur = 0;
let autostarted = false;

function fmtClock(s: number): string {
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${m}:${String(ss).padStart(2, "0")}`;
}

function renderLine(l: TraceLine, t0: number): HTMLElement {
  const el = document.createElement("span");
  el.className = "pl" + (l.k ? " milestone" : "");
  const stamp = document.createElement("span");
  stamp.className = "pl-t";
  stamp.textContent = fmtClock(t0);
  const msg = document.createElement("span");
  msg.className = l.c;
  msg.textContent = l.m;
  el.append(stamp, msg);
  return el;
}

function buildTimes() {
  if (!data) return;
  lineStart = [];
  let acc = 0;
  for (const l of data.lines) {
    lineStart.push(acc);
    acc += l.t;
  }
  totalDur = acc;
}

function renderUpTo(n: number, animateLast = false) {
  if (!data) return;
  body.textContent = "";
  const frag = document.createDocumentFragment();
  for (let i = 0; i < n; i++) {
    const el = renderLine(data.lines[i], lineStart[i]);
    if (animateLast && i === n - 1 && !reduced) {
      el.style.opacity = "0";
      el.style.transform = "translateY(6px)";
      frag.append(el);
      requestAnimationFrame(() => {
        el.style.transition = "opacity 0.25s, transform 0.25s";
        el.style.opacity = "1";
        el.style.transform = "none";
      });
    } else {
      frag.append(el);
    }
  }
  body.append(frag);
  body.scrollTop = body.scrollHeight;
  playhead = n;
  counter.textContent = `${n}/${data.lines.length}`;
  scrub.value = String(Math.round((n / data.lines.length) * 1000));
  const t = n > 0 ? lineStart[n - 1] + data.lines[n - 1].t : 0;
  clock.textContent = fmtClock(t);
  finalPanel.hidden = n < data.lines.length;
}

function tick(ts: number) {
  if (!playing || !data) return;
  const dt = (ts - lastTs) / 1000;
  lastTs = ts;
  const curT = playhead > 0 ? lineStart[playhead - 1] + data.lines[playhead - 1].t : 0;
  // advance: lines whose start time falls within current play time
  let target = curT + dt * 2.2; // 2.2x playback for a brisk but readable pace
  let n = playhead;
  while (n < data.lines.length && lineStart[n] <= target) n++;
  if (n > playhead) {
    // append incrementally instead of full re-render
    const frag = document.createDocumentFragment();
    for (let i = playhead; i < n; i++) {
      const el = renderLine(data.lines[i], lineStart[i]);
      frag.append(el);
      if (!reduced && i === n - 1) {
        el.style.opacity = "0";
        el.style.transform = "translateY(6px)";
        requestAnimationFrame(() => {
          el.style.transition = "opacity 0.25s, transform 0.25s";
          el.style.opacity = "1";
          el.style.transform = "none";
        });
      }
    }
    body.append(frag);
    body.scrollTop = body.scrollHeight;
    playhead = n;
    counter.textContent = `${n}/${data.lines.length}`;
    scrub.value = String(Math.round((n / data.lines.length) * 1000));
    clock.textContent = fmtClock(lineStart[n - 1] + data.lines[n - 1].t);
  }
  if (playhead >= data.lines.length) {
    pause();
    finalPanel.hidden = false;
    track("player_complete", { run: data.key });
    return;
  }
  raf = requestAnimationFrame(tick);
}

function play() {
  if (!data || playing) return;
  if (playhead >= data.lines.length) renderUpTo(0);
  playing = true;
  track("player_play", { run: data.key });
  ppIcon.textContent = "❚❚";
  lastTs = performance.now();
  raf = requestAnimationFrame(tick);
}

function pause() {
  playing = false;
  ppIcon.textContent = "▶";
  cancelAnimationFrame(raf);
}

async function loadRun(key: string) {
  pause();
  // show the skeleton while fetching; clear any previous run's content
  ppBtn.disabled = true;
  body.textContent = "";
  let d = cache.get(key);
  if (!d) {
    const skel = document.getElementById("player-skeleton");
    if (skel) skel.style.display = "";
    try {
      const res = await fetch(`/traces/${key}.json`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      d = (await res.json()) as TraceData;
      cache.set(key, d);
    } catch (err) {
      // fetch failed — show a clear error instead of a blank terminal
      if (skel) skel.style.display = "none";
      body.textContent = `[error] failed to load trace: ${err instanceof Error ? err.message : "unknown"}. Refresh or try another run.`;
      body.className = "term-body player-body";
      return;
    }
  }
  // hide skeleton now that data is ready
  const skel = document.getElementById("player-skeleton");
  if (skel) skel.style.display = "none";
  data = d;
  buildTimes();
  metaSrc.textContent = d.source;
  metaStats.textContent = `${d.n_runs} rounds · ${d.n_llm} model calls · ${d.wall_min} min wall · ${d.n_events} raw events`;
  termTitle.textContent = `trace.jsonl — ${d.key} run · real events, failures preserved`;
  finalBadge.textContent = "FINAL TALLY";
  finalText.textContent = `${d.final.supported} supported · ${d.final.falsified} falsified · ${d.final.inconclusive} inconclusive · judge ${d.judge}`;
  finalPanel.hidden = true;
  renderUpTo(0);
  ppBtn.disabled = false;
}

function switchRun(key: string) {
  document.querySelectorAll<HTMLButtonElement>(".run-btn").forEach((b) => {
    const on = b.dataset.run === key;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
  });
  if (data && data.key !== key) track("player_switch", { run: key });
  void loadRun(key);
}

/* wiring */
const runBtns = Array.from(document.querySelectorAll<HTMLButtonElement>(".run-btn"));
runBtns.forEach((b) => {
  b.addEventListener("click", () => switchRun(b.dataset.run!));
});

ppBtn.addEventListener("click", () => (playing ? pause() : play()));

let scrubTracked = false;
scrub.addEventListener("input", () => {
  if (!data) return;
  pause();
  if (!scrubTracked && data) {
    scrubTracked = true;
    track("player_scrub", { run: data.key });
  }
  const frac = Number(scrub.value) / 1000;
  const n = Math.round(frac * data.lines.length);
  renderUpTo(n);
});

body.addEventListener("keydown", (e) => {
  if (e.key === " " || e.key === "Spacebar") {
    e.preventDefault();
    if (playing) pause();
    else play();
  }
});

/* one-time autoplay when the player scrolls into view */
if (!reduced) {
  const io = new IntersectionObserver(
    (entries) => {
      if (entries[0].isIntersecting && !autostarted && data) {
        autostarted = true;
        setTimeout(play, 400);
        io.disconnect();
      }
    },
    { threshold: 0.4 }
  );
  io.observe(body.closest(".player") ?? body);
}

/* initial run: first tab, or ?run= override (used by /papers/<slug> pages) */
const params = new URLSearchParams(location.search);
const initial =
  params.get("run") && runBtns.some((b) => b.dataset.run === params.get("run"))
    ? (params.get("run") as string)
    : runBtns[0]?.dataset.run;
if (initial) void loadRun(initial);
