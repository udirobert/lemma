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

export {};
