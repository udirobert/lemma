/* Frontend sensors for the interest gates (see HACKATHON.md roadmap):
   - GitHub star count on the waitlist badge (cached 10 min)
   - Plausible custom events on every CTA / form submit
   - Netlify form AJAX submission with inline confirmation        */

declare global {
  interface Window {
    plausible?: (
      name: string,
      opts?: { props?: Record<string, string>; callback?: () => void }
    ) => void;
  }
}

const REPO = "udirobert/lemma";

function track(name: string, props: Record<string, string> = {}) {
  window.plausible?.(name, { props });
}

/* ---------- github stars ---------- */
async function loadStars() {
  const el = document.getElementById("gh-stars");
  if (!el) return;
  try {
    const cached = localStorage.getItem("gh-stars");
    if (cached) {
      const { v, ts } = JSON.parse(cached) as { v: number; ts: number };
      if (Date.now() - ts < 10 * 60 * 1000) {
        el.textContent = String(v);
        return;
      }
    }
    const res = await fetch(`https://api.github.com/repos/${REPO}`);
    if (!res.ok) throw new Error(String(res.status));
    const data = (await res.json()) as { stargazers_count: number };
    el.textContent = String(data.stargazers_count);
    localStorage.setItem(
      "gh-stars",
      JSON.stringify({ v: data.stargazers_count, ts: Date.now() })
    );
  } catch {
    el.textContent = "star";
  }
}
void loadStars();

/* ---------- CTA tracking ---------- */
document.addEventListener("click", (e) => {
  const cta = (e.target as HTMLElement).closest?.("[data-cta]");
  if (cta) track("cta", { cta: (cta as HTMLElement).dataset.cta ?? "?" });
});

/* ---------- waitlist form (Netlify, AJAX) ---------- */
const form = document.querySelector<HTMLFormElement>("form[name='lemma-interest']");
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const interest = String(fd.get("interest") ?? "waitlist");
    track("form_submit", { interest });

    const btn = form.querySelector<HTMLButtonElement>(".wl-btn");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "submitting…";
    }
    try {
      const res = await fetch("/", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams(
          Object.fromEntries(fd.entries()) as Record<string, string>
        ).toString(),
      });
      if (!res.ok) throw new Error(String(res.status));
      form.innerHTML =
        '<p class="wl-thanks">you\'re on the list — we\'ll write when the gate opens. ' +
        (interest === "request-audit"
          ? "if you named a paper, it just moved up the queue."
          : "") +
        "</p>";
      track("form_success", { interest });
    } catch {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "get early access";
      }
      form.setAttribute("data-netlify", "true");
      form.submit(); // fall back to a native Netlify POST
    }
  });
}

/* ---------- recorded-audit lookup ---------- */
/* When a user types an arXiv/source id into the "which paper" field, look it
   up in the recorded audit bundle and show that paper's actual claim titles
   (linking into the Audit Room). No extraction is simulated — a paper with
   no recorded audit says so and points at the local CLI. */
import bundle from "./data/audit-room.json";
import { findPaper, type Bundle } from "./roomModel";

(function recordedLookup() {
  const input = document.getElementById("paper-input") as HTMLInputElement | null;
  const preview = document.getElementById("claim-preview");
  const status = document.getElementById("preview-status");
  const list = document.getElementById("preview-claims");
  if (!input || !preview || !status || !list) return;

  const data = bundle as unknown as Bundle;
  const ID_RE = /^(?:arxiv:?)?\d{4}\.\d{4,5}(v\d+)?$/i;
  let debounceT: ReturnType<typeof setTimeout> | undefined;

  function show(id: string) {
    const paper = findPaper(data, id) ?? findPaper(data, `arxiv-${id.replace(/^arxiv:?/i, "")}`);
    preview.hidden = false;
    list.replaceChildren();
    if (!paper) {
      status.textContent =
        "no recorded audit for this id — new papers run via ./lemma audit <arxiv-id>";
      return;
    }
    status.textContent = `${paper.claims.length} recorded claims · ${paper.source_label}`;
    for (const c of paper.claims) {
      const el = document.createElement("a");
      el.className = "wl-claim";
      el.href = `/room/?paper=${paper.slug}&claim=${c.id}`;
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = c.id;
      const txt = document.createElement("span");
      txt.className = "txt";
      txt.textContent = c.title;
      el.append(tag, txt);
      list.append(el);
    }
    track("recorded_lookup", { id, paper: paper.slug });
  }

  input.addEventListener("input", () => {
    clearTimeout(debounceT);
    const v = input.value.trim();
    debounceT = setTimeout(() => {
      if (ID_RE.test(v)) show(v);
    }, 400);
  });
})();

export {};
