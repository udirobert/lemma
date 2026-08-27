/* Paper-detail + registry motion: section reveals, staggered claim rows/stats/
   figures/paper-cards, count-up stats, and figure load fade-in with shimmer.
   IntersectionObserver + CSS transitions — no GSAP, so these pages stay light
   while matching the landing's easing vocabulary (power3.out ≈ this bezier).
   Reveals use the `translate` property (not `transform`) so native hover
   transforms on cards/cells never conflict with the reveal.
   Progressive: hidden states only exist under html.pd-motion, which this
   script adds. No JS → everything visible. Reduced-motion → no animation. */

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* sequential header items — revealed top-to-bottom on page load.
   Both page families listed; each page only matches its own. */
const HEADER_ORDER = [
  ".pd-meta-row",
  ".pd-header h1",
  ".pd-authors",
  ".pd-blurb",
  ".pd-links",
  ".registry-header h1",
  ".registry-sub",
];

/* scroll-revealed singles (section headings, player) */
const SINGLES = [
  ".pd-claims h2",
  ".pd-figures h2",
  ".pd-trace h2",
  ".pd-trace-sub",
  ".pd-trace .player",
];

/* staggered groups — items reveal in sibling order */
const GROUPS = [".pd-stat", ".claim-row", ".figure-cell", ".paper-card"];

const REVEAL_EASE = "cubic-bezier(0.22, 0.61, 0.36, 1)"; // ≈ power3.out

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/* count-up for .pd-stat b — mirrors the landing's counter (power2.out, 1.6s) */
function countUp(el: HTMLElement) {
  const target = Number(el.textContent);
  if (!Number.isFinite(target)) return;
  const dur = 1600;
  const start = performance.now();
  const tick = (now: number) => {
    const p = Math.min(1, (now - start) / dur);
    el.textContent = String(Math.round(target * easeOutCubic(p)));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* position of el among its same-class siblings (for stagger delay) */
function groupIndex(el: Element): number {
  const cls = GROUPS.find((c) => el.matches(c));
  if (!cls || !el.parentElement) return 0;
  const sibs = el.parentElement.querySelectorAll(":scope > " + cls);
  return Array.from(sibs).indexOf(el);
}

function init() {
  if (reduced) return; // static page — everything visible, numbers final
  document.documentElement.classList.add("pd-motion");

  const reveal: HTMLElement[] = [];
  const delayFor = new Map<HTMLElement, number>();

  /* header cascade: delay by position among the items this page actually has */
  const headerItems = HEADER_ORDER.map((sel) => document.querySelector<HTMLElement>(sel)).filter(
    (el): el is HTMLElement => el !== null,
  );
  headerItems.forEach((el, i) => {
    reveal.push(el);
    delayFor.set(el, i * 80);
  });
  for (const sel of SINGLES) {
    document.querySelectorAll<HTMLElement>(sel).forEach((el) => {
      reveal.push(el);
      delayFor.set(el, 0);
    });
  }
  for (const sel of GROUPS) {
    document.querySelectorAll<HTMLElement>(sel).forEach((el) => {
      reveal.push(el);
      delayFor.set(el, Math.min(groupIndex(el) * 60, 360));
    });
  }

  /* figures: fade the image in when loaded; shimmer until then */
  document.querySelectorAll<HTMLImageElement>(".figure-cell img").forEach((img) => {
    const cell = img.closest(".figure-cell");
    const done = () => {
      img.classList.add("is-loaded");
      cell?.classList.add("fig-done");
    };
    if (img.complete && img.naturalWidth > 0) done();
    else {
      img.addEventListener("load", done, { once: true });
      img.addEventListener("error", done, { once: true });
    }
  });

  const show = (el: HTMLElement) => {
    el.style.setProperty("--pd-delay", `${delayFor.get(el) ?? 0}ms`);
    el.classList.add("is-in");
    if (el.classList.contains("pd-stat")) {
      const b = el.querySelector("b");
      if (b) countUp(b);
    }
    /* once the reveal transition finishes, hand the element back to its
       native styles (pd-done) so hover transforms/transitions on cards
       and figure cells behave exactly as unstyled */
    el.addEventListener(
      "transitionend",
      (ev) => {
        if (ev.target !== el) return;
        el.classList.remove("is-in");
        el.classList.add("pd-done");
      },
      { once: true },
    );
  };

  try {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          show(e.target as HTMLElement);
          io.unobserve(e.target);
        }
      },
      { threshold: 0.15 },
    );
    reveal.forEach((el) => io.observe(el));
  } catch {
    /* no IntersectionObserver — reveal everything immediately */
    reveal.forEach(show);
    document
      .querySelectorAll<HTMLElement>(".figure-cell img")
      .forEach((img) => img.classList.add("is-loaded"));
  }
}

init();
