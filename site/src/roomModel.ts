export interface Checks {
  state: "passed" | "failed" | "inconclusive" | "not_evaluated";
  control: "passed" | "failed" | "missing";
  problems: string[];
}

export interface Summary {
  claim_id?: string;
  status?: string;
  metrics?: Record<string, unknown>;
  notes?: string;
}

export interface TextArtifact {
  name: string;
  text: string;
  truncated: boolean;
  sha256?: string;
}

export interface ScriptArtifact extends TextArtifact {
  sha256: string;
}

export interface AttemptFeedback {
  human: string;
  generated: string;
  binding: string;
}

export interface Attempt {
  id: string;
  number: number;
  run_id: string | null;
  source: string;
  record_type: "legacy" | "versioned" | string;
  created_at: string | null;
  script: ScriptArtifact | null;
  criterion: string | null;
  criterion_sha256: string | null;
  feedback: AttemptFeedback;
  summary: Summary | null;
  checks: Checks;
  exit_code: number | null;
  wall_s: number | null;
  stdout: TextArtifact | null;
  stderr: TextArtifact | null;
  figures: { name: string; url: string; sha256: string }[];
  weave_url: string | null;
  execution: string;
}

export interface Claim {
  id: string;
  title: string;
  statement: string;
  evidence_in_paper: string;
  criterion: string;
  test_plan: string;
  compute: string;
  testable: boolean;
  status: string;
  xylo_label: string;
  xylo_color: string;
  summary: Summary | null;
  checks: Checks;
  source: string;
  candidate_summary: Summary | null;
  reference_summary: Summary | null;
  reference_checked: boolean;
  reference_disagreement: boolean;
  final_attempt_id: string | null;
  attempts: Attempt[];
  current_feedback: {
    human: TextArtifact | null;
    generated: TextArtifact | null;
    binding: string;
  };
  current_figures: { name: string; url: string; sha256: string }[];
}

export interface Improvement {
  ts: string;
  targets: string[];
  before: Record<string, string>;
  after: Record<string, string>;
  changed: string[];
  round_id?: string;
}

export interface RoomPaper {
  slug: string;
  paper_id: string;
  title: string;
  source_label: string;
  source_url: string | null;
  role: string;
  links: { logbook: string | null; dataset: string | null; paper: string | null };
  scope_note?: string | null;
  claims: Claim[];
  improvements: Improvement[];
}

export interface Bundle {
  schema_version: number;
  mode: string;
  captured_at: string;
  papers: RoomPaper[];
}

export function claimKey(paperSlug: string, claimId: string): string {
  return JSON.stringify([paperSlug, claimId]);
}

export type BarState = "on" | "dim" | "fail";

export function claimBarState(claim: Pick<Claim, "status">): BarState {
  if (claim.status === "supported") return "on";
  if (claim.status === "falsified") return "fail";
  return "dim";
}

export const PENTATONIC = [
  261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25, 783.99, 880.0,
  1046.5, 1174.66, 1318.51, 1567.98, 1760.0, 2093.0, 2349.32, 2637.02,
];

export function claimBarFreq(index: number): number {
  return PENTATONIC[index % PENTATONIC.length];
}

export function claimBarHeight(index: number, count: number): number {
  return Math.round(34 + (index / Math.max(1, count - 1)) * 66);
}

export function sourceLabel(source: string | null | undefined): string {
  switch (source) {
    case "agent_generated":
      return "Agent-generated";
    case "human_guided":
      return "Human-guided";
    case "reviewer_reference":
      return "Reviewer reference";
    default:
      return "Provenance not recorded";
  }
}

export function statusLabel(status: string | null | undefined): string {
  switch (status) {
    case "supported":
      return "supported";
    case "falsified":
      return "falsified";
    case "inconclusive":
      return "inconclusive";
    case "not_audited":
      return "not audited";
    default:
      return "unknown";
  }
}

export function checkStateLabel(checks: Checks | null | undefined): string {
  if (!checks) return "not evaluated";
  switch (checks.state) {
    case "passed":
      return "summary checks passed";
    case "failed":
      return "summary checks failed";
    case "inconclusive":
      return "checks inconclusive";
    default:
      return "not evaluated";
  }
}

export function controlLabel(checks: Checks | null | undefined): string {
  if (!checks) return "control not evaluated";
  switch (checks.control) {
    case "passed":
      return "control passed";
    case "failed":
      return "control failed";
    default:
      return "control missing";
  }
}

export function feedbackEntries(
  feedback: AttemptFeedback,
): { label: string; text: string }[] {
  const out: { label: string; text: string }[] = [];
  if (feedback.human) out.push({ label: "Human feedback", text: feedback.human });
  if (feedback.generated)
    out.push({ label: "Generated advisory notes", text: feedback.generated });
  return out;
}

export function findPaper(bundle: Bundle, query: string): RoomPaper | null {
  const q = query.trim().toLowerCase();
  if (!q) return null;
  const exact = bundle.papers.find(
    (p) => p.slug.toLowerCase() === q || p.paper_id.toLowerCase() === q,
  );
  if (exact) return exact;
  const matches = bundle.papers.filter((p) =>
    [p.title, p.source_label, p.paper_id, p.slug].some((f) =>
      f.toLowerCase().includes(q),
    ),
  );
  return matches.length === 1 ? matches[0] : null;
}

export function searchPapers(bundle: Bundle, query: string): RoomPaper[] {
  const q = query.trim().toLowerCase();
  if (!q) return bundle.papers;
  return bundle.papers.filter((p) =>
    [p.title, p.source_label, p.paper_id, p.slug].some((f) =>
      f.toLowerCase().includes(q),
    ),
  );
}

export function defaultAttempt(claim: Claim): Attempt | null {
  if (!claim.attempts.length) return null;
  const pinned = claim.attempts.find((a) => a.id === claim.final_attempt_id);
  return pinned ?? claim.attempts[claim.attempts.length - 1];
}

export interface OutcomeView {
  status: string;
  summary: Summary | null;
  checks: Checks;
  source: string;
  attempt: Attempt | null;
}

export function outcomeView(claim: Claim, attempt: Attempt | null): OutcomeView {
  if (attempt) {
    return {
      status: attempt.summary?.status ?? "output_unavailable",
      summary: attempt.summary,
      checks: attempt.checks,
      source: attempt.source,
      attempt,
    };
  }
  return {
    status: claim.status,
    summary: claim.summary,
    checks: claim.checks,
    source: claim.source,
    attempt: null,
  };
}

export type CriterionBadge = "same" | "changed" | "unpinned";

export function criterionBadge(before: Attempt | null, after: Attempt | null): CriterionBadge {
  if (
    before?.record_type === "versioned" &&
    after?.record_type === "versioned" &&
    before.criterion_sha256 &&
    after.criterion_sha256
  ) {
    return before.criterion_sha256 === after.criterion_sha256
      ? "same"
      : "changed";
  }
  return "unpinned";
}

export interface DiffLine {
  kind: "same" | "added" | "removed";
  text: string;
}

export function diffLines(before: string, after: string): DiffLine[] {
  const a = before.split("\n");
  const b = after.split("\n");
  let pre = 0;
  const maxPre = Math.min(a.length, b.length);
  while (pre < maxPre && a[pre] === b[pre]) pre++;
  let suf = 0;
  while (
    suf < a.length - pre &&
    suf < b.length - pre &&
    a[a.length - 1 - suf] === b[b.length - 1 - suf]
  ) {
    suf++;
  }
  const out: DiffLine[] = [];
  for (let i = 0; i < pre; i++) out.push({ kind: "same", text: a[i] });
  for (let i = pre; i < a.length - suf; i++)
    out.push({ kind: "removed", text: a[i] });
  for (let i = pre; i < b.length - suf; i++)
    out.push({ kind: "added", text: b[i] });
  for (let i = a.length - suf; i < a.length; i++)
    out.push({ kind: "same", text: a[i] });
  return out;
}

export function hasChangedRegion(lines: DiffLine[]): boolean {
  return lines.some((l) => l.kind !== "same");
}
