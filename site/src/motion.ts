/* Lemma landing motion engine.
   Lenis smooth scroll + GSAP ScrollTrigger: pinned hero exit, beat reveals,
   animated counters, a scroll-accumulating audit trace, and an easeReverse
   clip-path menu. Honors prefers-reduced-motion (static, no pins). */

import gsap from "gsap";
import ScrollTrigger from "gsap/ScrollTrigger";
import Lenis from "lenis";

gsap.registerPlugin(ScrollTrigger);

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- smooth scroll ---------- */
let lenis: Lenis | null = null;
if (!reduced) {
  lenis = new Lenis({ lerp: 0.09, smoothWheel: true });
  lenis.on("scroll", ScrollTrigger.update);
  gsap.ticker.add((time) => lenis!.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);
  document.documentElement.classList.add("js-motion");
}

function scrollTo(hash: string) {
  const el = document.querySelector(hash);
  if (!el) return;
  if (lenis) lenis.scrollTo(el as HTMLElement, { offset: -20 });
  else (el as HTMLElement).scrollIntoView({ behavior: "smooth" });
}

/* ---------- xylophone audio ---------- */
let actx: AudioContext | null = null;
const readout = document.getElementById("xylo-readout");

function strike(bar: HTMLElement) {
  const freq = Number(bar.dataset.freq);
  const label = bar.dataset.label ?? "";
  actx = actx ?? new AudioContext();
  if (actx.state === "suspended") void actx.resume();
  const t = actx.currentTime;
  for (const [mult, gain0, dur, type] of [
    [1, 0.2, 1.5, "triangle"],
    [2.01, 0.055, 0.7, "sine"],
    [3.98, 0.02, 0.35, "sine"],
  ] as const) {
    const osc = actx.createOscillator();
    const g = actx.createGain();
    osc.type = type;
    osc.frequency.value = freq * mult;
    g.gain.setValueAtTime(gain0, t);
    g.gain.exponentialRampToValueAtTime(0.0008, t + dur);
    osc.connect(g).connect(actx.destination);
    osc.start(t);
    osc.stop(t + dur + 0.05);
  }
  const state = bar.classList.contains("dim")
    ? "inconclusive — and we said so"
    : "supported";
  if (readout) readout.textContent = `claim ${label} · ${state}`;
  if (!reduced) {
    gsap.fromTo(
      bar,
      { y: -14, scale: 1.06 },
      { y: 0, scale: 1, duration: 0.55, ease: "elastic.out(1.1, 0.42)" }
    );
  }
}

for (const bar of document.querySelectorAll<HTMLElement>(".bar")) {
  bar.addEventListener("click", () => strike(bar));
}

/* ---------- bar entrance ---------- */
const bars = Array.from(document.querySelectorAll<HTMLElement>(".bar"));
if (!reduced && bars.length) {
  gsap.from(bars, {
    y: 60,
    rotateX: 45,
    opacity: 0,
    duration: 0.9,
    ease: "back.out(1.6)",
    stagger: 0.055,
    delay: 0.25,
    transformOrigin: "bottom center",
  });
}

/* ---------- pinned hero: row -> DNA helix -> spin -> exit ---------- */
if (!reduced) {
  const xylo = document.querySelector<HTMLElement>(".hero .xylo");
  const heroBars = xylo ? Array.from(xylo.querySelectorAll<HTMLElement>(".bar")) : [];
  if (xylo && heroBars.length) {
    const cx = xylo.clientWidth / 2;
    const cy = xylo.clientHeight / 2;
    const n = heroBars.length;
    const spacing = 30;
    // layout-based (transform-agnostic) natural centers, relative to .xylo
    const targets = heroBars.map((bar, i) => ({
      x: cx - (bar.offsetLeft + bar.offsetWidth / 2),
      y: (i - (n - 1) / 2) * spacing + cy - (bar.offsetTop + bar.offsetHeight / 2),
      rotY: i * (360 / n),
    }));

    const helix = gsap.timeline({
      scrollTrigger: {
        trigger: ".hero",
        start: "top top",
        end: "+=300%",
        scrub: 0.6,
        pin: true,
        anticipatePin: 1,
      },
    });

    // phase 0: clear the stage
    helix.to(".hero-copy", { yPercent: -18, opacity: 0, ease: "power2.in", duration: 0.5 }, 0);
    helix.to(".hero .xylo-readout", { opacity: 0, duration: 0.3 }, 0);
    helix.to(".hero .scroll-cue", { opacity: 0, duration: 0.2 }, 0);

    // phase 1: bars fly into a single-turn helix of horizontal rungs
    heroBars.forEach((bar, i) => {
      helix.to(
        bar,
        {
          x: targets[i].x,
          y: targets[i].y,
          rotateY: targets[i].rotY,
          rotationZ: 90,
          transformOrigin: "50% 50%",
          ease: "power2.inOut",
          duration: 1.0,
        },
        0.35 + i * 0.04
      );
    });

    // phase 2: spin the helix one full turn (scroll-driven conveyor)
    heroBars.forEach((bar, i) => {
      helix.to(bar, { rotateY: targets[i].rotY + 360, ease: "none", duration: 1.3 }, 1.7);
    });

    // phase 3: exit into the story
    helix.to(".hero .xylo", { opacity: 0, scale: 0.82, y: -40, ease: "power2.in", duration: 0.5 }, 2.7);
  }
}

/* ---------- beat reveals ---------- */
if (!reduced) {
  for (const beat of document.querySelectorAll(".beat")) {
    gsap.from(beat.querySelectorAll(".beat-n, h3, p, .big-stat, .receipts"), {
      y: 42,
      opacity: 0,
      duration: 0.85,
      ease: "power3.out",
      stagger: 0.09,
      scrollTrigger: { trigger: beat, start: "top 72%" },
    });
  }
}

/* ---------- counters ---------- */
function countUp(el: HTMLElement, target: number, decimals = 0) {
  if (reduced) {
    el.textContent = target.toFixed(decimals);
    return;
  }
  const obj = { v: 0 };
  gsap.to(obj, {
    v: target,
    duration: 1.6,
    ease: "power2.out",
    onUpdate: () => {
      el.textContent = obj.v.toFixed(decimals);
    },
    scrollTrigger: { trigger: el, start: "top 80%" },
  });
}
for (const el of document.querySelectorAll<HTMLElement>("[data-count]")) {
  countUp(el, Number(el.dataset.count), Number(el.dataset.decimals ?? 0));
}

/* ---------- pinned trace accumulation ---------- */
const lines = Array.from(document.querySelectorAll<HTMLElement>(".tl"));
if (!reduced && lines.length) {
  gsap.set(lines, { opacity: 0, y: 8 });
  const tl = gsap.timeline({
    scrollTrigger: {
      trigger: ".trace",
      start: "top 12%",
      end: "bottom 45%",
      scrub: 0.4,
      pin: ".trace-pin",
      pinnedContainer: ".trace",
    },
  });
  for (const line of lines) {
    tl.to(line, { opacity: 1, y: 0, duration: 0.5, ease: "power2.out" }, "+=0.28");
  }
  tl.to({}, { duration: 0.6 });
}

/* ---------- artifacts reveals ---------- */
if (!reduced) {
  gsap.from(".link-row", {
    y: 26,
    opacity: 0,
    duration: 0.7,
    ease: "power3.out",
    stagger: 0.08,
    scrollTrigger: { trigger: ".artifacts", start: "top 75%" },
  });
}

/* ---------- clip menu (GSAP easeReverse) ---------- */
const menuBtn = document.querySelector<HTMLButtonElement>(".menu-btn");
const menuPanel = document.querySelector<HTMLElement>(".menu-panel");
const menuItems = gsap.utils.toArray<HTMLElement>(".menu-panel li");
let menuOpen = false;

const openTl = gsap.timeline({ paused: true });
if (!reduced && menuPanel) {
  openTl
    .set(menuPanel, { visibility: "visible" })
    .to(
      menuPanel,
      {
        clipPath: "circle(142% at calc(100% - 55px) calc(100% - 55px))",
        duration: 0.62,
        ease: "expo.inOut",
        easeReverse: "power3.out",
        onReverseComplete: () => {
          if (menuPanel) menuPanel.style.visibility = "hidden";
        },
      },
      0
    )
    .fromTo(
      ".menu-panel .menu-sub",
      { opacity: 0, y: 12 },
      { opacity: 1, y: 0, duration: 0.4, ease: "power3.out" },
      0.18
    )
    .fromTo(
      menuItems,
      { opacity: 0, y: 40 },
      {
        opacity: 1,
        y: 0,
        duration: 0.55,
        ease: "expo.out",
        easeReverse: "power3.in",
        stagger: 0.07,
      },
      0.22
    )
    .to(
      ".menu-btn .bars-icon",
      { rotate: 90, duration: 0.4, ease: "power2.inOut" },
      0
    );
}

function toggleMenu(force?: boolean) {
  menuOpen = force ?? !menuOpen;
  menuPanel?.classList.toggle("open", menuOpen);
  if (reduced) {
    if (menuPanel) {
      menuPanel.style.visibility = "visible";
      menuPanel.style.clipPath = menuOpen
        ? "circle(142% at calc(100% - 55px) calc(100% - 55px))"
        : "circle(0% at calc(100% - 55px) calc(100% - 55px))";
      if (!menuOpen) menuPanel.style.visibility = "hidden";
    }
    return;
  }
  if (menuOpen) openTl.timeScale(1).play();
  else openTl.timeScale(1.35).reverse();
}

menuBtn?.addEventListener("click", () => toggleMenu());
for (const a of document.querySelectorAll<HTMLAnchorElement>(".menu-panel a")) {
  a.addEventListener("click", (e) => {
    const href = a.getAttribute("href");
    if (href?.startsWith("#")) {
      e.preventDefault();
      toggleMenu(false);
      setTimeout(() => scrollTo(href), reduced ? 0 : 250);
    } else {
      toggleMenu(false);
    }
  });
}
addEventListener("keydown", (e) => {
  if (e.key === "Escape" && menuOpen) toggleMenu(false);
});

/* nav pills in hero copy (if present) use the same scrolling */
for (const a of document.querySelectorAll<HTMLAnchorElement>("a[data-scroll]")) {
  a.addEventListener("click", (e) => {
    const href = a.getAttribute("href");
    if (href?.startsWith("#")) {
      e.preventDefault();
      scrollTo(href);
    }
  });
}
