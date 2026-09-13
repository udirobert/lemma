import rawBundle from "./data/audit-room.json";
import { playStrike } from "./audio";
import {
  checkStateLabel,
  claimBarFreq,
  claimBarHeight,
  claimBarState,
  claimKey,
  controlLabel,
  criterionBadge,
  defaultAttempt,
  diffLines,
  feedbackEntries,
  findPaper,
  hasChangedRegion,
  outcomeView,
  searchPapers,
  sourceLabel,
  statusLabel,
  type Attempt,
  type Bundle,
  type Claim,
  type Checks,
  type RoomPaper,
  type Summary,
  type TextArtifact,
} from "./roomModel";

const bundle = rawBundle as unknown as Bundle;

type Tab = "evidence" | "compare" | "script";

interface LiveSession {
  available: boolean;
  reason: string | null;
  token?: string;
  allowed?: { paper_slug: string; claim_id: string; attempt_id: string; script_sha256: string }[];
  limits?: { wall_s: number; cpu_s: number };
}

interface RerunJob {
  job_id: string;
  state: "queued" | "running" | "completed" | "failed";
  events: { stage: string; message: string; ts: string }[];
  attempt: Attempt | null;
  error: string | null;
  origin?: { paper_slug: string; claim_id: string; source_attempt_id: string };
}

const state = {
  paper: null as RoomPaper | null,
  claim: null as Claim | null,
  query: "",
  suggestions: [] as RoomPaper[],
  attemptId: null as string | null,
  beforeId: null as string | null,
  afterId: null as string | null,
  tab: "evidence" as Tab,
  present: false,
  mobilePanel: "evidence" as "claims" | "evidence" | "receipt",
  live: null as LiveSession | null,
  liveChecked: false,
  liveError: null as string | null,
  rerun: null as RerunJob | null,
  liveAttempts: new Map<string, Attempt[]>(),
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const root = $("room-root");
const claimsPanel = $("claims-panel");
const evidencePanel = $("evidence-panel");
const receiptPanel = $("receipt-panel");
const emptyEl = $("room-empty");
const grid = $("room-grid");
const dialog = $("rerun-dialog") as unknown as HTMLDialogElement;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function badge(cls: string, text: string): HTMLSpanElement {
  return el("span", `badge ${cls}`, text);
}

function statusBadge(status: string | null | undefined): HTMLSpanElement {
  const s = status ?? "unknown";
  const cls = ["supported", "falsified", "inconclusive"].includes(s)
    ? s
    : "neutral";
  return badge(cls, statusLabel(s));
}

function checkBadge(checks: Checks | null | undefined): HTMLSpanElement {
  if (!checks || checks.state === "not_evaluated")
    return badge("neutral", "not evaluated");
  const cls =
    checks.state === "passed" ? "passed" : checks.state === "failed" ? "failed" : "inconclusive";
  return badge(cls, checkStateLabel(checks));
}

function controlBadge(checks: Checks | null | undefined): HTMLSpanElement {
  if (!checks) return badge("neutral", "control not evaluated");
  const cls =
    checks.control === "passed" ? "passed" : checks.control === "failed" ? "failed" : "missing";
  return badge(cls, controlLabel(checks));
}

function fmtMetric(v: unknown): string {
  if (typeof v === "number")
    return Number.isFinite(v) ? String(Number(v.toPrecision(6))) : String(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

function claimAttempts(claim: Claim): Attempt[] {
  const key = state.paper ? claimKey(state.paper.slug, claim.id) : "";
  return [...claim.attempts, ...(state.liveAttempts.get(key) ?? [])];
}

function selectedAttempt(): Attempt | null {
  if (!state.claim) return null;
  return claimAttempts(state.claim).find((a) => a.id === state.attemptId) ?? null;
}

function readUrl() {
  const p = new URLSearchParams(location.search);
  const pq = p.get("paper") ?? p.get("q") ?? "";
  state.query = pq;
  state.suggestions = [];
  state.paper = pq
    ? findPaper(bundle, pq)
    : (findPaper(bundle, "icl-bayesian") ?? bundle.papers[0] ?? null);
  if (pq && !state.paper) state.suggestions = searchPapers(bundle, pq);
  const cq = p.get("claim");
  state.claim =
    state.paper?.claims.find((c) => c.id === cq) ??
    state.paper?.claims.find((c) => c.id === "C6") ??
    state.paper?.claims[0] ??
    null;
  const attempts = state.claim ? claimAttempts(state.claim) : [];
  const sel = p.get("attempt");
  state.attemptId =
    sel && attempts.some((a) => a.id === sel)
      ? sel
      : (state.claim ? (defaultAttempt(state.claim)?.id ?? null) : null);
  const b = p.get("before");
  const a = p.get("after");
  state.beforeId =
    b && attempts.some((x) => x.id === b) ? b : (attempts[0]?.id ?? null);
  state.afterId =
    a && attempts.some((x) => x.id === a)
      ? a
      : (attempts[attempts.length - 1]?.id ?? null);
  const t = p.get("tab");
  state.tab = t === "compare" || t === "script" ? t : "evidence";
  state.present = p.get("present") === "1";
}

function writeUrl(push = true) {
  const p = new URLSearchParams();
  if (state.paper) p.set("paper", state.paper.slug);
  else if (state.query) p.set("q", state.query);
  if (state.claim) p.set("claim", state.claim.id);
  if (state.attemptId) p.set("attempt", state.attemptId);
  if (state.beforeId) p.set("before", state.beforeId);
  if (state.afterId) p.set("after", state.afterId);
  if (state.tab !== "evidence") p.set("tab", state.tab);
  if (state.present) p.set("present", "1");
  const url = `${location.pathname}?${p.toString()}`;
  if (push) history.pushState(null, "", url);
  else history.replaceState(null, "", url);
}

function renderHeader() {
  root.classList.remove(...PAPER_CLASSES);
  if (state.paper) root.classList.add(paperColorClass());
  const options = $("paper-options");
  options.textContent = "";
  for (const p of bundle.papers) {
    const opt = document.createElement("option");
    opt.value = p.slug;
    opt.label = `${p.title} — ${p.source_label}`;
    options.append(opt);
  }
  const title = $("paper-title");
  title.textContent = state.paper ? state.paper.title : "";
  title.title = state.paper ? `${state.paper.title} — ${state.paper.source_label}` : "";
  const link = $("source-link") as HTMLAnchorElement;
  const url = state.paper?.source_url ?? state.paper?.links.paper ?? null;
  if (url) {
    link.href = url;
    link.hidden = false;
  } else {
    link.hidden = true;
  }
  const pill = $("mode-pill");
  if (state.liveChecked && state.live?.available) {
    pill.textContent = "RECORDED + LOCAL RUNNER";
    pill.classList.add("live");
  } else {
    pill.textContent = "RECORDED EVIDENCE";
    pill.classList.remove("live");
  }
  ($("present-toggle") as HTMLButtonElement).setAttribute(
    "aria-pressed",
    String(state.present),
  );
}

const PAPER_CLASSES = ["pc-blue", "pc-magenta", "pc-teal", "pc-gold", "pc-violet"];

function paperColorClass(): string {
  const c = state.paper?.claims[0]?.xylo_color ?? "c-blue";
  return PAPER_CLASSES.includes(`p${c}`) ? `p${c}` : "pc-blue";
}

function renderClaims() {
  claimsPanel.textContent = "";
  if (!state.paper) return;
  claimsPanel.append(el("h2", "", `claims · ${state.paper.slug}`));

  const claims = state.paper.claims;
  const n = claims.length;
  const rail = el("div", "claim-rail");
  rail.setAttribute("role", "group");
  rail.setAttribute(
    "aria-label",
    "Claim instrument — one bar per claim; strike a bar to select it",
  );
  claims.forEach((c, i) => {
    const st = claimBarState(c);
    const bar = el("button", `bar ${st} ${c.xylo_color || "c-blue"}`);
    bar.type = "button";
    bar.style.setProperty("--h", `${claimBarHeight(i, n)}px`);
    bar.style.setProperty("--i", String(i));
    bar.dataset.freq = String(claimBarFreq(i));
    bar.dataset.label = c.xylo_label || c.id;
    bar.dataset.state = st;
    bar.dataset.cid = c.id;
    bar.setAttribute(
      "aria-label",
      `${c.id} · ${c.xylo_label || c.id} — ${statusLabel(c.status)}`,
    );
    bar.setAttribute("aria-pressed", String(c === state.claim));
    if (c === state.claim) bar.classList.add("selected");
    bar.append(el("span", "bar-label", c.xylo_label || c.id));
    bar.addEventListener("click", () => {
      try {
        playStrike(Number(bar.dataset.freq));
      } catch {
      }
      selectClaim(c.id);
    });
    rail.append(bar);
  });
  claimsPanel.append(rail);
  claimsPanel.append(
    el(
      "p",
      "claim-rail-readout",
      state.claim
        ? `claim ${state.claim.xylo_label || state.claim.id} · ${statusLabel(state.claim.status)}`
        : `${n} claims · strike a bar`,
    ),
  );

  const list = el("ul", "claim-list");
  list.setAttribute("role", "listbox");
  list.setAttribute("aria-label", "Claims");
  state.paper.claims.forEach((c, i) => {
    const li = el("li", "claim-item");
    const btn = el("button");
    btn.type = "button";
    btn.setAttribute("role", "option");
    btn.setAttribute("aria-selected", String(c === state.claim));
    btn.dataset.cid = c.id;
    const top = el("span", "cid");
    top.append(el("span", "", c.id), statusBadge(c.status));
    btn.append(top, el("span", "ct", c.title));
    btn.addEventListener("click", () => selectClaim(c.id));
    btn.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      e.preventDefault();
      const next = e.key === "ArrowDown" ? i + 1 : i - 1;
      const target = state.paper?.claims[next];
      if (target) {
        selectClaim(target.id);
        claimsPanel
          .querySelector<HTMLButtonElement>(`.claim-list [data-cid="${target.id}"]`)
          ?.focus();
      }
    });
    li.append(btn);
    list.append(li);
  });
  claimsPanel.append(list);
  if (state.paper.scope_note)
    claimsPanel.append(el("p", "small muted", state.paper.scope_note));
}

function kv(term: string, node: Node | string): [HTMLElement, HTMLElement] {
  const dt = el("dt", "", term);
  const dd = el("dd");
  if (typeof node === "string") dd.textContent = node;
  else dd.append(node);
  return [dt, dd];
}

function metricsGrid(summary: Summary | null): HTMLElement | null {
  const metrics = summary?.metrics ?? {};
  const keys = Object.keys(metrics);
  if (!keys.length) return null;
  const gridEl = el("div", "metrics-grid");
  for (const k of keys) {
    const cell = el("div", "metric");
    cell.append(el("div", "mk", k), el("div", "mv", fmtMetric(metrics[k])));
    gridEl.append(cell);
  }
  return gridEl;
}

function summaryBlock(summary: Summary | null): HTMLElement {
  const wrap = el("div");
  if (!summary) {
    wrap.append(el("p", "small muted", "Output unavailable — no recorded summary."));
    return wrap;
  }
  if (summary.notes) wrap.append(el("div", "notes-block", summary.notes));
  const m = metricsGrid(summary);
  if (m) wrap.append(m);
  return wrap;
}

function metricsDisclosure(summary: Summary | null): HTMLElement | null {
  const m = metricsGrid(summary);
  if (!m) return null;
  const d = el("details", "improvement-note");
  d.append(el("summary", "", "All recorded metrics"));
  d.append(m);
  return d;
}

function figureCard(f: { name: string; url: string; sha256: string }): HTMLElement {
  const fig = el("figure");
  const img = el("img");
  img.src = f.url;
  img.alt = f.name;
  img.loading = "lazy";
  const cap = el("figcaption");
  cap.append(
    el("span", "fig-name", f.name),
    el("span", "fig-sha", `sha ${f.sha256.slice(0, 8)}`),
  );
  fig.append(img, cap);
  return fig;
}

function figuresBlock(claim: Claim, attempt: Attempt | null): HTMLElement {
  const wrap = el("div");
  const pinned = attempt?.figures ?? [];
  if (pinned.length) {
    wrap.append(el("h3", "", `figures from ${attempt?.id}`));
    const row = el("div", "room-figs");
    for (const f of pinned) row.append(figureCard(f));
    wrap.append(row);
    return wrap;
  }
  if (claim.current_figures.length) {
    wrap.append(
      el("h3", "", "current figures"),
      el(
        "p",
        "small muted",
        "Current figure snapshot — not pinned to a historical attempt.",
      ),
    );
    const row = el("div", "room-figs");
    for (const f of claim.current_figures) row.append(figureCard(f));
    wrap.append(row);
  }
  return wrap;
}

function renderEvidenceTab(body: HTMLElement) {
  const claim = state.claim;
  if (!claim) return;
  const attempt = selectedAttempt();
  const view = outcomeView(claim, attempt);

  const head = el("div", "claim-head");
  const h2 = el("h2");
  h2.append(
    el("span", "claim-id", claim.id),
    document.createTextNode(claim.title),
  );
  head.append(h2);
  if (attempt) head.append(badge("neutral", `attempt ${attempt.number} selected`));
  head.append(
    statusBadge(view.status === "output_unavailable" ? null : view.status),
    checkBadge(view.checks),
    controlBadge(view.checks),
  );
  body.append(head);

  if (view.summary?.notes) {
    body.append(el("div", "notes-block", view.summary.notes));
  } else if (attempt && !view.summary) {
    body.append(el("p", "small muted", "Output unavailable — no recorded summary for this attempt."));
  }
  if (view.checks.problems.length) {
    const ul = el("ul", "small problems-list");
    for (const p of view.checks.problems) ul.append(el("li", "", p));
    body.append(ul);
  }
  const md = metricsDisclosure(view.summary);
  if (md) body.append(md);
  body.append(figuresBlock(claim, attempt));

  const setup = el("details", "improvement-note");
  setup.append(el("summary", "", "Claim and test setup"));
  const meta = el("dl", "room-kv");
  const [t1, d1] = kv("statement", claim.statement);
  const [t2, d2] = kv("source locator", claim.evidence_in_paper || "—");
  const critWrap = el("div");
  critWrap.append(el("span", "", claim.criterion));
  critWrap.append(
    el(
      "div",
      "criterion-note",
      "Extracted test criterion · not independently verified against the paper",
    ),
  );
  const [t3, d3] = kv("test criterion", critWrap);
  const [t4, d4] = kv("test plan", claim.test_plan || "—");
  const [t5, d5] = kv("outcome source", sourceLabel(view.source));
  meta.append(t1, d1, t2, d2, t3, d3, t4, d4, t5, d5);
  setup.append(meta);
  body.append(setup);

  const fin = el("details", "improvement-note");
  fin.append(
    el(
      "summary",
      "",
      `Final recorded outcome — ${statusLabel(claim.status)} (claim of record, never replaced by reruns)`,
    ),
  );
  fin.append(summaryBlock(claim.summary));
  body.append(fin);

  if (claim.reference_disagreement) {
    const d = el("div", "disagreement");
    d.append(
      el("h4", "", "Reference disagreement — candidate and reference verdicts differ"),
    );
    const cand = el("details");
    cand.append(el("summary", "", "candidate summary"));
    cand.append(summaryBlock(claim.candidate_summary));
    const ref = el("details");
    ref.append(el("summary", "", "reviewer reference summary"));
    ref.append(summaryBlock(claim.reference_summary));
    d.append(cand, ref);
    body.append(d);
  } else if (claim.reference_checked && claim.reference_summary) {
    const d = el("details", "improvement-note");
    d.append(
      el("summary", "", "reviewer reference summary (checked — verdicts agree)"),
    );
    d.append(summaryBlock(claim.reference_summary));
    body.append(d);
  }

  const imps = (state.paper?.improvements ?? []).filter((i) =>
    i.targets.includes(claim.id),
  );
  if (imps.length) {
    const box = el("div", "improvement-note");
    box.append(
      el(
        "p",
        "small",
        `This claim was included in ${imps.length} improvement round(s).`,
      ),
    );
    for (const i of imps) {
      box.append(
        el(
          "p",
          "mono small muted",
          `${i.ts} · ${i.before[claim.id] ?? "?"} → ${i.after[claim.id] ?? "?"}` +
            (i.changed.includes(claim.id) ? " · changed" : ""),
        ),
      );
    }
    box.append(
      el(
        "p",
        "small muted",
        "Legacy round records are unlinked — they do not prove which scripts produced these statuses.",
      ),
    );
    body.append(box);
  }

  const fb = claim.current_feedback;
  if (fb.human || fb.generated) {
    const d = el("details", "improvement-note");
    d.append(
      el(
        "summary",
        "",
        "Current advisory notes — not linked to these historical attempts",
      ),
    );
    for (const art of [fb.human, fb.generated]) {
      if (!art) continue;
      d.append(el("h4", "mono small", art.name));
      d.append(el("div", "notes-block", art.text));
      if (art.truncated)
        d.append(el("p", "truncated-flag", "Truncated in export."));
    }
    body.append(d);
  }
}

function renderHistory(host: HTMLElement) {
  const claim = state.claim;
  if (!claim) return;
  host.append(el("h3", "", "attempt history"));
  const strip = el("div", "history-strip");
  strip.setAttribute("role", "listbox");
  strip.setAttribute("aria-label", "Attempts");
  for (const a of claimAttempts(claim)) {
    const chip = el("button", "attempt-chip");
    chip.type = "button";
    chip.setAttribute("role", "option");
    chip.setAttribute("aria-selected", String(a.id === state.attemptId));
    const isLive =
      state.paper &&
      state.liveAttempts
        .get(claimKey(state.paper.slug, claim.id))
        ?.some((x) => x.id === a.id);
    if (isLive) chip.classList.add("live-rerun");
    chip.append(
      el("span", "an", `attempt ${a.number}${isLive ? " · LIVE RERUN" : ""}`),
      el("span", "as", sourceLabel(a.source)),
      el(
        "span",
        "as",
        a.summary?.status
          ? `${statusLabel(a.summary.status)} · ${checkStateLabel(a.checks)}`
          : a.execution === "output_unavailable"
            ? "output unavailable"
            : "no summary",
      ),
    );
    chip.addEventListener("click", () => {
      state.attemptId = a.id;
      writeUrl();
      renderAll();
    });
    strip.append(chip);
  }
  host.append(strip);
}

function attemptOptionLabel(a: Attempt): string {
  return `${a.number} · ${sourceLabel(a.source)} · ${a.summary?.status ?? a.execution}`;
}

function attemptCard(title: string, a: Attempt | null): HTMLElement {
  const card = el("div", "compare-card");
  card.append(el("h4", "", title));
  if (!a) {
    card.append(el("p", "small muted", "No attempt selected."));
    return card;
  }
  const row = el("div", "compare-badges");
  row.append(
    el("span", "mono small", `#${a.number} · ${sourceLabel(a.source)}`),
    statusBadge(a.summary?.status),
    controlBadge(a.checks),
    checkBadge(a.checks),
  );
  card.append(row);
  if (a.summary?.notes) card.append(el("p", "small", a.summary.notes));
  const dl = el("dl", "room-kv");
  dl.append(...kv("wall", a.wall_s === null ? "—" : `${a.wall_s}s`));
  card.append(dl);
  if (a.checks.problems.length) {
    const ul = el("ul", "small");
    for (const p of a.checks.problems) ul.append(el("li", "", p));
    card.append(ul);
  }
  const det = el("details", "output-details");
  det.append(el("summary", "", "record details"));
  const raw = el("dl", "room-kv");
  raw.append(
    ...kv("attempt id", a.id),
    ...kv("record", a.record_type),
    ...kv("execution", a.execution),
    ...kv("run_id", a.run_id ?? "—"),
    ...kv("exit code", a.exit_code === null ? "—" : String(a.exit_code)),
    ...kv("script sha256", a.script?.sha256 ?? "—"),
  );
  det.append(raw);
  card.append(det);
  return card;
}

function renderCompareTab(body: HTMLElement) {
  const claim = state.claim;
  if (!claim) return;
  const attempts = claimAttempts(claim);
  const before = attempts.find((a) => a.id === state.beforeId) ?? null;
  const after = attempts.find((a) => a.id === state.afterId) ?? null;
  const sel = selectedAttempt();
  if (sel)
    body.append(badge("neutral", `attempt ${sel.number} selected in evidence`));

  const selects = el("div", "compare-selects");
  for (const [label, key] of [
    ["Before", "beforeId"],
    ["After", "afterId"],
  ] as const) {
    const lab = el("label", "", label);
    const selEl = el("select");
    selEl.setAttribute("aria-label", `${label} attempt`);
    for (const a of attempts) {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = attemptOptionLabel(a);
      opt.selected = a.id === state[key];
      selEl.append(opt);
    }
    selEl.addEventListener("change", () => {
      state[key] = selEl.value;
      writeUrl();
      renderAll();
    });
    lab.append(selEl);
    selects.append(lab);
  }
  body.append(selects);

  const cb = criterionBadge(before, after);
  const cbadge =
    cb === "same"
      ? badge("passed", "Same recorded criterion")
      : cb === "changed"
        ? badge("failed", "Criterion changed — review required")
        : badge("neutral", "Historical criterion not pinned");
  body.append(cbadge);
  body.append(
    el(
      "p",
      "small muted",
      "An identical recorded input does not establish that the run followed it.",
    ),
  );

  const cols = el("div", "compare-cols");
  cols.append(attemptCard("before", before), attemptCard("after", after));
  body.append(cols);

  if (before || after) {
    const afterPinned =
      after?.feedback.binding === "attempt_input" &&
      (after.feedback.human || after.feedback.generated);
    if (afterPinned) {
      const d = el("details", "improvement-note");
      d.append(
        el("summary", "", "Recorded input feedback — after attempt"),
      );
      for (const f of feedbackEntries(after!.feedback)) {
        d.append(el("h4", "mono small", f.label));
        d.append(el("div", "notes-block", f.text));
      }
      body.append(d);
    }
    const legacySide = [before, after].some(
      (a) => a && a.record_type !== "versioned",
    );
    if (legacySide) {
      body.append(
        el(
          "p",
          "small muted",
          "One side is a legacy record — its input feedback and criterion were not pinned at run time.",
        ),
      );
      const fb = claim.current_feedback;
      if (!afterPinned && (fb.human || fb.generated)) {
        const d = el("details", "improvement-note");
        d.append(
          el(
            "summary",
            "",
            "Current advisory notes — not linked to these historical attempts",
          ),
        );
        for (const art of [fb.human, fb.generated]) {
          if (art) d.append(el("div", "notes-block", art.text));
        }
        body.append(d);
      }
    }
  }

  const diff = el("div", "diff-block");
  const head = el("div", "diff-head", "Changed region — every line retained, not a minimal diff");
  diff.append(head);
  const lines = el("div", "diff-lines");
  const aText = before?.script?.text ?? "";
  const bText = after?.script?.text ?? "";
  if (!before?.script && !after?.script) {
    lines.append(el("div", "dl", "No recorded scripts for these attempts."));
  } else {
    const rows = diffLines(aText, bText);
    if (!hasChangedRegion(rows))
      lines.append(el("div", "dl", "Scripts are identical."));
    else for (const r of rows) lines.append(el("div", `dl ${r.kind}`, r.text));
    if (before?.script?.truncated || after?.script?.truncated) {
      diff.append(
        el("p", "truncated-flag", "One or both recorded scripts are truncated in export."),
      );
    }
  }
  diff.append(lines);
  body.append(diff);
}

function artifactDetails(title: string, art: TextArtifact | null): HTMLElement {
  const d = el("details", "output-details");
  const s = el("summary", "", title);
  d.append(s);
  if (!art) {
    d.append(el("p", "small muted", "No recorded output for this attempt."));
    return d;
  }
  d.append(el("pre", "code-block", art.text || "(empty)"));
  if (art.truncated) d.append(el("p", "truncated-flag", "Truncated in export."));
  return d;
}

function renderScriptTab(body: HTMLElement) {
  const claim = state.claim;
  if (!claim) return;
  const attempt = selectedAttempt();
  if (!attempt) {
    body.append(el("p", "muted", "Select an attempt to inspect its script."));
    return;
  }
  const head = el("div", "claim-head");
  head.append(el("h3", "", `script · attempt ${attempt.number}`));
  if (attempt.script) {
    head.append(
      badge("neutral", `sha256 ${attempt.script.sha256.slice(0, 12)}…`),
    );
  }
  const copyBtn = el("button", "room-btn", "Copy script");
  copyBtn.type = "button";
  const copyMsg = el("span", "small muted");
  copyBtn.addEventListener("click", () => {
    const text = attempt.script?.text ?? "";
    if (!navigator.clipboard) {
      copyMsg.textContent = "clipboard unavailable — select the text manually";
      return;
    }
    navigator.clipboard
      .writeText(text)
      .then(() => {
        copyMsg.textContent = "copied";
      })
      .catch(() => {
        copyMsg.textContent = "copy failed — select the text manually";
      });
  });
  copyBtn.disabled = !attempt.script;
  head.append(copyBtn, copyMsg);
  body.append(head);
  if (attempt.script) {
    body.append(el("pre", "code-block", attempt.script.text));
    if (attempt.script.truncated)
      body.append(el("p", "truncated-flag", "Recorded script is truncated in export."));
  } else {
    body.append(el("p", "muted", "No recorded script for this attempt."));
  }
  body.append(
    artifactDetails("stdout", attempt.stdout),
    artifactDetails("stderr", attempt.stderr),
  );
  const entries = feedbackEntries(attempt.feedback);
  if (attempt.feedback.binding === "attempt_input" && entries.length) {
    const d = el("details", "improvement-note");
    d.append(el("summary", "", "Recorded input feedback"));
    for (const f of entries) {
      d.append(el("h4", "mono small", f.label));
      d.append(el("div", "notes-block", f.text));
    }
    body.append(d);
  }
}

function renderReceipt() {
  receiptPanel.textContent = "";
  const claim = state.claim;
  if (!claim || !state.paper) return;
  const view = outcomeView(claim, selectedAttempt());
  receiptPanel.append(el("h2", "", "receipt"));
  const list = el("div", "receipt-list");
  const rows: [string, string][] = [
    ["paper", state.paper.source_label],
    ["captured", bundle.captured_at.slice(0, 10)],
    ["final verdict", statusLabel(claim.status)],
    ["selected", view.attempt ? `attempt ${view.attempt.number}` : "final"],
    ["sel. status", statusLabel(view.status === "output_unavailable" ? null : view.status)],
    ["sel. control", controlLabel(view.checks)],
    ["provenance", sourceLabel(view.source)],
    ["attempts", String(claimAttempts(claim).length)],
    ["reference", claim.reference_checked ? "checked" : "not validated"],
  ];
  for (const [k, v] of rows) {
    const r = el("div", "receipt-row");
    r.append(el("span", "rk", k), el("span", "rv", v));
    list.append(r);
  }
  receiptPanel.append(list);

  const attempt = selectedAttempt();
  if (attempt?.weave_url) {
    const a = el("a", "small", "Open Weave trace");
    a.href = attempt.weave_url;
    a.target = "_blank";
    a.rel = "noopener";
    receiptPanel.append(a);
  } else {
    receiptPanel.append(el("p", "small muted", "No call link recorded"));
  }

  renderLiveControls(receiptPanel, attempt);
  renderHistory(receiptPanel);
}

const isLocal =
  location.hostname === "localhost" || location.hostname === "127.0.0.1";

async function checkLive() {
  if (!isLocal) {
    state.liveChecked = true;
    state.live = { available: false, reason: "static hosting" };
    renderHeader();
    renderReceipt();
    return;
  }
  try {
    const res = await fetch("/api/session", {
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.live = (await res.json()) as LiveSession;
  } catch (e) {
    state.live = { available: false, reason: "no local runner" };
    state.liveError = e instanceof Error ? e.message : String(e);
  }
  state.liveChecked = true;
  renderHeader();
  renderReceipt();
}

function rerunAllowed(attempt: Attempt | null): boolean {
  if (!state.live?.available || !attempt || !state.paper || !state.claim)
    return false;
  if (!attempt.script) return false;
  return (state.live.allowed ?? []).some(
    (m) =>
      m.paper_slug === state.paper!.slug &&
      m.claim_id === state.claim!.id &&
      m.attempt_id === attempt.id &&
      m.script_sha256 === attempt.script!.sha256,
  );
}

function renderLiveControls(host: HTMLElement, attempt: Attempt | null) {
  const box = el("div", "improvement-note");
  if (!isLocal || !state.live?.available) {
    box.append(
      el(
        "p",
        "small muted",
        `Recorded mode · live reruns require the local runner${
          state.live?.reason ? ` (${state.live.reason})` : ""
        }${state.liveError ? ` — ${state.liveError}` : ""}`,
      ),
    );
    if (isLocal) {
      const cmd = el("p", "mono small muted");
      cmd.append(
        document.createTextNode("npm --prefix site run build"),
        el("br"),
        document.createTextNode("./lemma room"),
      );
      box.append(cmd);
    }
    host.append(box);
    return;
  }
  const btn = el("button", "room-btn primary", "Review & rerun");
  btn.type = "button";
  const allowed = rerunAllowed(attempt);
  const busy = state.rerun?.state === "queued" || state.rerun?.state === "running";
  btn.disabled = !allowed || busy;
  if (!allowed)
    box.append(
      el(
        "p",
        "small muted",
        "This attempt is not in the local rerun manifest (exact script digest required).",
      ),
    );
  btn.addEventListener("click", () => openRerunDialog(attempt));
  box.append(btn);
  if (state.rerun) {
    const j = state.rerun;
    const origin = j.origin
      ? ` · origin ${j.origin.paper_slug}/${j.origin.claim_id} attempt ${j.origin.source_attempt_id}`
      : "";
    box.append(el("p", "mono small", `rerun ${j.job_id} · ${j.state}${origin}`));
    if (j.events.length) {
      const ev = el("div", "rerun-events");
      for (const e of j.events) {
        const row = el("div", "ev");
        row.append(el("span", "st", `[${e.stage}] `), document.createTextNode(e.message));
        ev.append(row);
      }
      box.append(ev);
    }
    if (j.error) box.append(el("p", "small", `error: ${j.error}`));
  }
  host.append(box);
}

function openRerunDialog(attempt: Attempt | null) {
  if (!attempt || !state.claim || !state.paper) return;
  dialog.textContent = "";
  dialog.append(el("h3", "", "Review & rerun"));
  dialog.append(
    el(
      "p",
      "small",
      "Same script and seed · no new LLM call · not independent replication",
    ),
  );
  const dl = el("dl", "room-kv");
  dl.append(
    ...kv("claim", `${state.claim.id} — ${state.claim.title}`),
    ...kv("attempt", `${attempt.id} · #${attempt.number}`),
    ...kv(
      "script sha256",
      attempt.script ? attempt.script.sha256 : "no recorded script",
    ),
    ...kv(
      "limits",
      state.live?.limits
        ? `wall ${state.live.limits.wall_s}s · cpu ${state.live.limits.cpu_s}s`
        : "runner defaults",
    ),
  );
  dialog.append(dl);
  const actions = el("div", "dlg-actions");
  const cancel = el("button", "room-btn", "Cancel");
  cancel.type = "button";
  cancel.addEventListener("click", () => dialog.close());
  const run = el("button", "room-btn primary", "Run isolated check");
  run.type = "button";
  run.addEventListener("click", () => {
    dialog.close();
    void startRerun(attempt);
  });
  actions.append(cancel, run);
  dialog.append(actions);
  dialog.showModal();
}

let pollTimer: ReturnType<typeof setTimeout> | null = null;
let postInFlight = false;

async function startRerun(attempt: Attempt) {
  if (!state.live?.token || !state.paper || !state.claim || postInFlight) return;
  postInFlight = true;
  const origin = {
    paper_slug: state.paper.slug,
    claim_id: state.claim.id,
    source_attempt_id: attempt.id,
  };
  state.rerun = {
    job_id: "…",
    state: "queued",
    events: [],
    attempt: null,
    error: null,
    origin,
  };
  renderReceipt();
  try {
    const res = await fetch("/api/reruns", {
      method: "POST",
      signal: AbortSignal.timeout(10000),
      headers: {
        "Content-Type": "application/json",
        "X-Lemma-Token": state.live.token,
      },
      body: JSON.stringify({
        paper_slug: origin.paper_slug,
        claim_id: origin.claim_id,
        attempt_id: attempt.id,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const job = (await res.json()) as { job_id: string; state: string };
    state.rerun = {
      job_id: job.job_id,
      state: "queued",
      events: [],
      attempt: null,
      error: null,
      origin,
    };
    pollRerun(job.job_id, origin);
  } catch (e) {
    state.rerun = {
      job_id: "—",
      state: "failed",
      events: [],
      attempt: null,
      error: `rerun request failed: ${e instanceof Error ? e.message : String(e)}`,
      origin,
    };
  } finally {
    postInFlight = false;
  }
  renderReceipt();
}

function pollRerun(
  jobId: string,
  origin: { paper_slug: string; claim_id: string; source_attempt_id: string },
) {
  if (pollTimer) clearTimeout(pollTimer);
  const tick = async () => {
    try {
      const res = await fetch(`/api/reruns/${jobId}`, {
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const job = (await res.json()) as RerunJob;
      state.rerun = { ...job, origin };
      if (job.state === "completed" && job.attempt) {
        const key = claimKey(origin.paper_slug, origin.claim_id);
        const list = state.liveAttempts.get(key) ?? [];
        if (!list.some((a) => a.id === job.attempt!.id)) {
          list.push(job.attempt);
          state.liveAttempts.set(key, list);
        }
        const stillSelected =
          state.paper?.slug === origin.paper_slug &&
          state.claim?.id === origin.claim_id;
        if (stillSelected) {
          state.afterId = job.attempt.id;
          state.attemptId = job.attempt.id;
          state.tab = "compare";
          writeUrl();
        }
        renderAll();
        return;
      }
      if (job.state === "failed") {
        renderAll();
        return;
      }
      renderReceipt();
      pollTimer = setTimeout(() => void tick(), 1000);
    } catch (e) {
      state.rerun = {
        job_id: jobId,
        state: "failed",
        events: state.rerun?.events ?? [],
        attempt: null,
        error: `Connection lost; server job may still be running (${
          e instanceof Error ? e.message : String(e)
        })`,
        origin,
      };
      renderReceipt();
    }
  };
  pollTimer = setTimeout(() => void tick(), 1000);
}

function renderEvidence() {
  evidencePanel.textContent = "";
  const claim = state.claim;
  if (!claim) return;

  const tabs = el("div", "room-tabs");
  tabs.setAttribute("role", "tablist");
  const tabIds: Tab[] = ["evidence", "compare", "script"];
  tabIds.forEach((t, i) => {
    const b = el("button", "", t === "script" ? "Script" : t[0].toUpperCase() + t.slice(1));
    b.type = "button";
    b.id = `tab-${t}`;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(state.tab === t));
    b.setAttribute("aria-controls", "tab-panel");
    b.addEventListener("click", () => {
      state.tab = t;
      writeUrl();
      renderAll();
    });
    b.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      const next = tabIds[(i + (e.key === "ArrowRight" ? 1 : tabIds.length - 1)) % tabIds.length];
      state.tab = next;
      writeUrl();
      renderAll();
      evidencePanel.querySelector<HTMLButtonElement>(`#tab-${next}`)?.focus();
    });
    tabs.append(b);
  });
  evidencePanel.append(tabs);

  const body = el("div", "tab-body");
  body.id = "tab-panel";
  body.setAttribute("role", "tabpanel");
  body.setAttribute("aria-labelledby", `tab-${state.tab}`);
  if (state.tab === "compare") renderCompareTab(body);
  else if (state.tab === "script") renderScriptTab(body);
  else renderEvidenceTab(body);
  evidencePanel.append(body);
}

function renderEmpty(msg: string) {
  grid.hidden = true;
  emptyEl.hidden = false;
  emptyEl.textContent = "";
  emptyEl.append(el("p", "", msg));
  if (state.suggestions.length) {
    const ul = el("ul", "suggest-list");
    for (const p of state.suggestions) {
      const li = el("li");
      const a = el("a", "", `${p.title} — ${p.source_label}`);
      a.href = `/room/?paper=${p.slug}`;
      li.append(a);
      ul.append(li);
    }
    emptyEl.append(el("p", "small", "Did you mean:"), ul);
  }
  const code = el("code", "", "./lemma audit <arxiv-id> --jobs 2");
  emptyEl.append(el("br"), code);
}

function renderAll() {
  root.classList.toggle("presenting", state.present);
  document
    .querySelectorAll(".room-mobile-nav button")
    .forEach((b) =>
      b.setAttribute(
        "aria-selected",
        String((b as HTMLElement).dataset.panel === state.mobilePanel),
      ),
    );
  for (const [panel, name] of [
    [claimsPanel, "claims"],
    [evidencePanel, "evidence"],
    [receiptPanel, "receipt"],
  ] as const) {
    panel.classList.toggle("panel-hidden-mobile", state.mobilePanel !== name);
  }
  renderHeader();
  if (!state.paper || !state.claim) {
    renderEmpty("No recorded audit found. New-paper audits use the local CLI.");
    return;
  }
  grid.hidden = false;
  emptyEl.hidden = true;
  renderClaims();
  renderEvidence();
  renderReceipt();
}

function selectClaim(cid: string) {
  const claim = state.paper?.claims.find((c) => c.id === cid);
  if (!claim) return;
  state.claim = claim;
  const attempts = claimAttempts(claim);
  state.attemptId = defaultAttempt(claim)?.id ?? null;
  state.beforeId = attempts[0]?.id ?? null;
  state.afterId = attempts[attempts.length - 1]?.id ?? null;
  state.mobilePanel = "evidence";
  writeUrl();
  renderAll();
}

function setPresent(on: boolean) {
  state.present = on;
  if (on && state.claim && claimAttempts(state.claim).length >= 2)
    state.tab = "compare";
  writeUrl();
  renderAll();
}

function stepAttempt(dir: 1 | -1) {
  const claim = state.claim;
  if (!claim) return;
  const attempts = claimAttempts(claim);
  const i = attempts.findIndex((a) => a.id === state.attemptId);
  const next = attempts[i + dir];
  if (next) {
    state.attemptId = next.id;
    writeUrl();
    renderAll();
  }
}

function init() {
  const search = $("paper-search") as HTMLInputElement;
  search.addEventListener("change", () => {
    const q = search.value.trim();
    state.query = q;
    const found = findPaper(bundle, q);
    if (found) {
      state.paper = found;
      state.suggestions = [];
      state.claim =
        found.claims.find((c) => c.id === "C6") ?? found.claims[0] ?? null;
      const attempts = state.claim ? claimAttempts(state.claim) : [];
      state.attemptId = state.claim ? (defaultAttempt(state.claim)?.id ?? null) : null;
      state.beforeId = attempts[0]?.id ?? null;
      state.afterId = attempts[attempts.length - 1]?.id ?? null;
      writeUrl();
      renderAll();
    } else {
      state.suggestions = searchPapers(bundle, q);
      state.paper = null;
      state.claim = null;
      writeUrl();
      renderAll();
    }
  });

  document.querySelectorAll(".room-mobile-nav button").forEach((b) => {
    b.addEventListener("click", () => {
      state.mobilePanel = (b as HTMLElement).dataset.panel as typeof state.mobilePanel;
      renderAll();
    });
  });

  $("present-toggle").addEventListener("click", () => setPresent(!state.present));
  $("exit-present").addEventListener("click", () => setPresent(false));
  $("present-prev").addEventListener("click", () => stepAttempt(-1));
  $("present-next").addEventListener("click", () => stepAttempt(1));

  document.addEventListener("keydown", (e) => {
    const target = e.target as HTMLElement;
    const typing =
      target.tagName === "INPUT" ||
      target.tagName === "TEXTAREA" ||
      target.tagName === "SELECT" ||
      target.isContentEditable;
    if (e.key === "Escape" && state.present) {
      setPresent(false);
      return;
    }
    if (state.present && !typing) {
      if (e.key === "ArrowRight") stepAttempt(1);
      if (e.key === "ArrowLeft") stepAttempt(-1);
    }
  });

  window.addEventListener("popstate", () => {
    readUrl();
    renderAll();
  });

  readUrl();
  if (!location.search) writeUrl(false);
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
    root.classList.add("room-enter");
    setTimeout(() => root.classList.remove("room-enter"), 1500);
  }
  renderAll();
  void checkLive();
}

init();
