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

/* ---------- mini claim-extractor preview ---------- */
/* When a user types something that looks like an arXiv id into the "which
   paper" field, show a simulated extraction preview — 3 plausible claim
   strings appear one by one, giving a taste of what lemma does. This is
   purely client-side (no real fetch); it's a teaser, not a real audit. */
(function miniExtractor() {
  const input = document.getElementById("paper-input") as HTMLInputElement | null;
  const preview = document.getElementById("claim-preview");
  const status = document.getElementById("preview-status");
  const list = document.getElementById("preview-claims");
  if (!input || !preview || !status || !list) return;

  const ARXIV_RE = /^(?:arxiv:?)?\d{4}\.\d{4,5}(v\d+)?$/i;
  const SAMPLE_CLAIMS = [
    "the proposed method outperforms the baseline by ≥15% on the benchmark",
    "convergence is achieved within 10k training steps under all tested configs",
    "the learned representation is linearly separable across all probe tasks",
  ];
  let fired = false;
  let debounceT: ReturnType<typeof setTimeout> | undefined;

  function runPreview(id: string) {
    if (fired) return;
    fired = true;
    preview.hidden = false;
    status.textContent = `extracting claims from ${id}…`;
    list.innerHTML = "";
    let i = 0;
    const reveal = () => {
      if (i >= SAMPLE_CLAIMS.length) {
        status.textContent = `${SAMPLE_CLAIMS.length} claims extracted · these are illustrative — the real audit runs the numbers`;
        track("mini_extract", { id });
        return;
      }
      const el = document.createElement("div");
      el.className = "wl-claim";
      el.style.animationDelay = "0s";
      el.innerHTML = `<span class="tag">C${i + 1}</span><span class="txt">${SAMPLE_CLAIMS[i]}</span>`;
      list.append(el);
      i++;
      setTimeout(reveal, 700);
    };
    setTimeout(reveal, 600);
  }

  input.addEventListener("input", () => {
    clearTimeout(debounceT);
    const v = input.value.trim();
    debounceT = setTimeout(() => {
      if (ARXIV_RE.test(v)) runPreview(v);
    }, 400);
  });
})();

export {};
