import { describe, it, expect } from "vitest";
import {
  attemptNote,
  claimBarFreq,
  claimBarHeight,
  claimBarState,
  claimKey,
  criterionBadge,
  criterionGauge,
  diffLines,
  diffScore,
  feedbackEntries,
  findPaper,
  hasChangedRegion,
  outcomeView,
  searchPapers,
  sourceLabel,
  statusTally,
  controlLabel,
  PENTATONIC,
  type Attempt,
  type Bundle,
  type Claim,
} from "../src/roomModel";
import bundleJson from "../src/data/audit-room.json";

const bundle = bundleJson as unknown as Bundle;

function attempt(partial: Partial<Attempt>): Attempt {
  return {
    id: "a1",
    number: 1,
    run_id: null,
    source: "agent_generated",
    record_type: "versioned",
    created_at: null,
    script: null,
    criterion: null,
    criterion_sha256: null,
    feedback: { human: "", generated: "", binding: "unavailable" },
    summary: null,
    checks: { state: "not_evaluated", control: "missing", problems: [] },
    exit_code: null,
    wall_s: null,
    stdout: null,
    stderr: null,
    figures: [],
    weave_url: null,
    execution: "recorded",
    ...partial,
  };
}

function claim(partial: Partial<Claim>): Claim {
  return {
    id: "C1",
    title: "c",
    statement: "",
    evidence_in_paper: "",
    criterion: "",
    test_plan: "",
    compute: "",
    testable: true,
    status: "supported",
    xylo_label: "c1",
    xylo_color: "c-blue",
    summary: { status: "supported", metrics: { m: 1 }, notes: "final" },
    checks: { state: "passed", control: "passed", problems: [] },
    source: "agent_generated",
    candidate_summary: null,
    reference_summary: null,
    reference_checked: false,
    reference_disagreement: false,
    final_attempt_id: null,
    attempts: [],
    current_feedback: { human: null, generated: null, binding: "current_unversioned" },
    current_figures: [],
    ...partial,
  };
}

describe("diffLines", () => {
  it("marks identical scripts entirely same", () => {
    const d = diffLines("a\nb\nc", "a\nb\nc");
    expect(d.every((l) => l.kind === "same")).toBe(true);
    expect(hasChangedRegion(d)).toBe(false);
  });

  it("handles empty inputs", () => {
    expect(diffLines("", "")).toEqual([{ kind: "same", text: "" }]);
    const added = diffLines("", "x");
    expect(added).toContainEqual({ kind: "added", text: "x" });
  });

  it("reconstructs the original by dropping added lines and the new by dropping removed", () => {
    const before = "def f():\n    return 1\n\nprint(f())";
    const after = "def f():\n    return 2\n\nprint(f())\nprint('extra')";
    const d = diffLines(before, after);
    expect(d).toContainEqual({ kind: "removed", text: "    return 1" });
    expect(d).toContainEqual({ kind: "added", text: "    return 2" });
    expect(d).toContainEqual({ kind: "added", text: "print('extra')" });
    const reconstructedBefore = d
      .filter((l) => l.kind !== "added")
      .map((l) => l.text)
      .join("\n");
    const reconstructedAfter = d
      .filter((l) => l.kind !== "removed")
      .map((l) => l.text)
      .join("\n");
    expect(reconstructedBefore).toBe(before);
    expect(reconstructedAfter).toBe(after);
  });

  it("keeps common suffix after a middle change", () => {
    const d = diffLines("x\ny\nz", "x\nY\nz");
    expect(d[0]).toEqual({ kind: "same", text: "x" });
    expect(d[1]).toEqual({ kind: "removed", text: "y" });
    expect(d[2]).toEqual({ kind: "added", text: "Y" });
    expect(d[3]).toEqual({ kind: "same", text: "z" });
  });
});

describe("criterionBadge", () => {
  it("only reports same when both versioned pins exist and match", () => {
    const a = attempt({ criterion_sha256: "aa" });
    const b = attempt({ id: "b", criterion_sha256: "aa" });
    expect(criterionBadge(a, b)).toBe("same");
    expect(criterionBadge(a, attempt({ id: "b", criterion_sha256: "bb" }))).toBe(
      "changed",
    );
  });
  it("is unpinned for legacy or missing hashes", () => {
    const legacy = attempt({ record_type: "legacy", criterion_sha256: null });
    const versioned = attempt({ id: "b", criterion_sha256: "aa" });
    expect(criterionBadge(legacy, legacy)).toBe("unpinned");
    expect(criterionBadge(legacy, versioned)).toBe("unpinned");
    expect(criterionBadge(null, null)).toBe("unpinned");
  });
});

describe("findPaper", () => {
  it("matches by slug, paper_id, title and source label", () => {
    expect(findPaper(bundle, "icl-bayesian")?.slug).toBe("icl-bayesian");
    const p = bundle.papers[0];
    expect(findPaper(bundle, p.paper_id)?.slug).toBe(p.slug);
  });
  it("returns null for unmatched or ambiguous input", () => {
    expect(findPaper(bundle, "no such paper exists")).toBeNull();
    expect(findPaper(bundle, "")).toBeNull();
    expect(findPaper(bundle, "grokking")).toBeNull(); // matches two papers
  });
});

describe("labels", () => {
  it("does not rename unknown provenance even when scripts exist", () => {
    expect(sourceLabel("unknown")).toBe("Provenance not recorded");
    expect(sourceLabel(null)).toBe("Provenance not recorded");
    expect(sourceLabel("agent_generated")).toBe("Agent-generated");
    expect(sourceLabel("human_guided")).toBe("Human-guided");
    expect(sourceLabel("reviewer_reference")).toBe("Reviewer reference");
  });
  it("control labels reflect recorded state, never coerced strings", () => {
    expect(
      controlLabel({ state: "failed", control: "failed", problems: [] }),
    ).toBe("control failed");
    expect(
      controlLabel({ state: "passed", control: "passed", problems: [] }),
    ).toBe("control passed");
    expect(
      controlLabel({ state: "not_evaluated", control: "missing", problems: [] }),
    ).toBe("control missing");
  });
});

describe("feedbackEntries", () => {
  it("maps literal feedback strings to labelled entries", () => {
    expect(
      feedbackEntries({
        human: "use log-space",
        generated: "try a grid",
        binding: "attempt_input",
      }),
    ).toEqual([
      { label: "Human feedback", text: "use log-space" },
      { label: "Generated advisory notes", text: "try a grid" },
    ]);
    expect(
      feedbackEntries({ human: "", generated: "g", binding: "attempt_input" }),
    ).toEqual([{ label: "Generated advisory notes", text: "g" }]);
    expect(
      feedbackEntries({ human: "", generated: "", binding: "unavailable" }),
    ).toEqual([]);
  });
});

describe("claimKey", () => {
  it("composes paper slug and claim id", () => {
    const a = claimKey("icl-bayesian", "C6");
    const b = claimKey("grokking-ca", "C6");
    expect(a).not.toBe(b);
    expect(a).toBe(JSON.stringify(["icl-bayesian", "C6"]));
  });
});

describe("outcomeView", () => {
  it("returns the selected attempt outcome when one is selected", () => {
    const sel = attempt({
      id: "x",
      summary: { status: "falsified", metrics: { m: 0 }, notes: "sel note" },
      checks: { state: "failed", control: "failed", problems: ["bad"] },
      source: "reviewer_reference",
    });
    const c = claim({ attempts: [sel] });
    const v = outcomeView(c, sel);
    expect(v.status).toBe("falsified");
    expect(v.summary?.notes).toBe("sel note");
    expect(v.checks.state).toBe("failed");
    expect(v.checks.control).toBe("failed");
    expect(v.source).toBe("reviewer_reference");
  });
  it("returns the final claim outcome when nothing is selected", () => {
    const v = outcomeView(claim({}), null);
    expect(v.status).toBe("supported");
    expect(v.summary?.notes).toBe("final");
  });
  it("never falls back to final metrics when the selected attempt has no summary", () => {
    const sel = attempt({ id: "x", summary: null });
    const v = outcomeView(claim({ attempts: [sel] }), sel);
    expect(v.status).toBe("output_unavailable");
    expect(v.summary).toBeNull();
  });
});

describe("claim rail (instrument) model", () => {
  it("every bundle claim carries a xylo label and a palette color", () => {
    const palette = new Set(["c-blue", "c-magenta", "c-teal", "c-gold", "c-violet"]);
    for (const p of bundle.papers) {
      for (const c of p.claims) {
        expect(c.xylo_label.length).toBeGreaterThan(0);
        expect(palette.has(c.xylo_color)).toBe(true);
      }
    }
  });
  it("all claims in a paper share the paper's palette color", () => {
    for (const p of bundle.papers) {
      expect(new Set(p.claims.map((c) => c.xylo_color)).size).toBe(1);
    }
  });
  it("bar state derives from the authoritative claim status, not the xylo label", () => {
    expect(claimBarState(claim({ status: "supported" }))).toBe("on");
    expect(claimBarState(claim({ status: "falsified" }))).toBe("fail");
    expect(claimBarState(claim({ status: "inconclusive" }))).toBe("dim");
    expect(claimBarState(claim({ status: "not_audited" }))).toBe("dim");
    const icl = findPaper(bundle, "icl-bayesian");
    const c2 = icl?.claims.find((c) => c.id === "C2");
    expect(c2?.status).toBe("falsified");
    expect(c2 ? claimBarState(c2) : null).toBe("fail");
  });
  it("assigns pentatonic frequencies by index within the paper", () => {
    expect(claimBarFreq(0)).toBe(PENTATONIC[0]);
    expect(claimBarFreq(PENTATONIC.length)).toBe(PENTATONIC[0]);
    expect(claimBarFreq(5)).toBe(PENTATONIC[5]);
  });
  it("interpolates bar heights across the rail range", () => {
    expect(claimBarHeight(0, 6)).toBe(34);
    expect(claimBarHeight(5, 6)).toBe(100);
    expect(claimBarHeight(0, 1)).toBe(34);
  });
});

describe("criterionGauge", () => {
  it("parses a symbolic inequality against a matching metric key", () => {
    const g = criterionGauge("slope < -0.5", { slope: -0.8 });
    expect(g).toEqual({
      op: "<",
      threshold: -0.5,
      metricKey: "slope",
      value: -0.8,
      pass: true,
    });
    const miss = criterionGauge("slope < -0.5", { slope: -0.2 });
    expect(miss?.pass).toBe(false);
  });
  it("handles >= and the unicode ≤ ≥ forms", () => {
    expect(criterionGauge("rate >= 0.9", { rate: 0.95 })?.op).toBe(">=");
    expect(criterionGauge("rate ≥ 0.9", { rate: 0.95 })?.pass).toBe(true);
    expect(criterionGauge("rate ≥ 0.9", { rate: 0.8 })?.pass).toBe(false);
    expect(criterionGauge("err ≤ 0.01", { err: 0.01 })?.op).toBe("<=");
    expect(criterionGauge("err <= 0.01", { err: 0.02 })?.pass).toBe(false);
  });
  it("parses unambiguous worded comparators", () => {
    expect(criterionGauge("correlation is less than 0.5", { correlation: 0.4 })?.op).toBe("<");
    expect(criterionGauge("accuracy at least 0.9", { accuracy: 0.9 })?.pass).toBe(true);
    expect(criterionGauge("loss at most 2", { loss: 3 })?.op).toBe("<=");
  });
  it("matches snake_case metric keys to their phrase in the criterion", () => {
    const g = criterionGauge(
      "with core compute ratio ≥0.95.",
      { core_compute_ratio: 0.97, retain_compute_ratio_mean: 0.5 },
    );
    expect(g?.metricKey).toBe("core_compute_ratio");
    expect(g?.threshold).toBe(0.95);
    expect(g?.pass).toBe(true);
  });
  it("dedupes a repeated identical comparison", () => {
    const g = criterionGauge("slope ≥ 0.5 and slope ≥ 0.5", { slope: 0.6 });
    expect(g?.op).toBe(">=");
  });
  it("returns null when metrics are missing or non-numeric", () => {
    expect(criterionGauge("slope < -0.5", {})).toBeNull();
    expect(criterionGauge("slope < -0.5", { slope: "n/a" })).toBeNull();
    expect(criterionGauge("slope < -0.5", { slope: NaN })).toBeNull();
  });
  it("returns null for ambiguous or absent comparisons", () => {
    expect(criterionGauge("", { slope: -0.8 })).toBeNull();
    expect(
      criterionGauge("x > 0.5 and y < 3", { x: 1, y: 2 }),
    ).toBeNull();
    expect(
      criterionGauge("R2 greater than 0.8 and at least 0.1 higher", {
        r2: 0.9,
      }),
    ).toBeNull();
  });
  it("refuses thresholds that are coefficients of another term", () => {
    expect(
      criterionGauge("the difference is less than 0.01 * R(M)", {
        difference: 0.005,
      }),
    ).toBeNull();
    expect(criterionGauge("within 5% of baseline", { ratio: 1.01 })).toBeNull();
  });
  it("refuses comparisons embedded in expressions", () => {
    expect(
      criterionGauge(
        "P_grok(lambda_1>0) > P_grok(lambda_1=0)",
        { p_grok: 0.4 },
      ),
    ).toBeNull();
  });
  it("returns null when zero or several metric keys match", () => {
    expect(criterionGauge("slope < -0.5", { other: -0.8 })).toBeNull();
    expect(
      criterionGauge("the gap ratio < 0.5", { gap_ratio: 0.4, ratio: 0.4 }),
    ).toBeNull();
  });
});

describe("attemptNote", () => {
  it("maps check state onto note glyphs honestly", () => {
    expect(
      attemptNote(attempt({ checks: { state: "passed", control: "passed", problems: [] } })),
    ).toBe("lit");
    expect(
      attemptNote(
        attempt({ checks: { state: "inconclusive", control: "failed", problems: [] } }),
      ),
    ).toBe("dim");
    expect(
      attemptNote(
        attempt({ checks: { state: "not_evaluated", control: "missing", problems: [] } }),
      ),
    ).toBe("dim");
  });
  it("failed integrity is snapped, never merely dim", () => {
    const a = attempt({
      summary: { status: "supported", metrics: {}, notes: "" },
      checks: { state: "failed", control: "passed", problems: ["contradiction"] },
    });
    expect(attemptNote(a)).toBe("snapped");
  });
});

describe("statusTally", () => {
  it("summarises verdicts in a stable order", () => {
    const c = (status: string) => claim({ status });
    expect(
      statusTally([
        c("supported"),
        c("supported"),
        c("supported"),
        c("falsified"),
        c("inconclusive"),
        c("inconclusive"),
      ]),
    ).toBe("3 supported · 1 falsified · 2 inconclusive");
  });
  it("handles empty and unknown statuses", () => {
    expect(statusTally([])).toBe("no verdicts");
    expect(statusTally([claim({ status: "weird" })])).toBe("1 other");
    expect(statusTally([claim({ status: "not_audited" })])).toBe("1 not audited");
  });
});

describe("diffScore", () => {
  const mkScript = (n: number) =>
    Array.from({ length: n }, (_, i) => `line ${i + 1}`).join("\n");

  it("emits change headers before each changed run", () => {
    const rows = diffScore("a\nb\nc", "a\nB\nc");
    expect(rows[0]).toMatchObject({ kind: "same", beforeNo: 1, afterNo: 1 });
    expect(rows[1]).toEqual({ kind: "change", count: 2 });
    expect(rows[2]).toMatchObject({ kind: "removed", text: "b", beforeNo: 2 });
    expect(rows[3]).toMatchObject({ kind: "added", text: "B", afterNo: 2 });
    expect(rows[4]).toMatchObject({ kind: "same", beforeNo: 3, afterNo: 3 });
  });
  it("collapses long unchanged runs into an expandable elision row", () => {
    const before = `${mkScript(10)}\nold\n${mkScript(10)}`;
    const after = `${mkScript(10)}\nnew\n${mkScript(10)}`;
    const rows = diffScore(before, after, { context: 3 });
    const elides = rows.filter((r) => r.kind === "elide");
    expect(elides.length).toBe(2);
    expect(elides[0]).toMatchObject({ count: 4 });
    const changes = rows.filter((r) => r.kind === "change");
    expect(changes).toEqual([{ kind: "change", count: 2 }]);
    const firstSame = rows[0];
    expect(firstSame).toMatchObject({ kind: "same", beforeNo: 1, afterNo: 1 });
    const tailSame = rows[rows.length - 1];
    expect(tailSame).toMatchObject({ kind: "same", beforeNo: 21, afterNo: 21 });
  });
  it("expands an elision when its key is in the expanded set", () => {
    const text = mkScript(20);
    const collapsed = diffScore(text, `${text}\nextra`);
    const elide = collapsed.find((r) => r.kind === "elide");
    expect(elide).toBeDefined();
    const expanded = diffScore(text, `${text}\nextra`, {
      expanded: new Set([elide!.kind === "elide" ? elide!.key : ""]),
    });
    expect(expanded.some((r) => r.kind === "elide")).toBe(false);
    expect(expanded.filter((r) => r.kind === "same").length).toBe(20);
  });
  it("emits no change rows for identical scripts", () => {
    const rows = diffScore("a\nb", "a\nb");
    expect(rows.some((r) => r.kind === "change")).toBe(false);
    expect(rows.every((r) => r.kind === "same")).toBe(true);
  });
});

describe("defaults and search", () => {
  it("resolves icl-bayesian and its C6 claim", () => {
    const p = findPaper(bundle, "icl-bayesian") ?? bundle.papers[0];
    expect(p.slug).toBe("icl-bayesian");
    expect(p.claims.some((c) => c.id === "C6")).toBe(true);
  });
  it("searchPapers returns multiple matches for ambiguous queries", () => {
    const matches = searchPapers(bundle, "grokking");
    expect(matches.length).toBeGreaterThanOrEqual(2);
  });
});
