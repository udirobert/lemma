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
  if (bar.dataset.dragged) { delete bar.dataset.dragged; return; }
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
  const st = bar.dataset.state;
  const state =
    st === "on"
      ? "supported"
      : st === "fail"
        ? "falsified"
        : "unverified — and we said so";
  const nearestReadout =
    bar.closest(".xylo-scene, .xylo-mirror")?.querySelector<HTMLElement>(".xylo-readout") ?? readout;
  if (nearestReadout) nearestReadout.textContent = `claim ${label} · ${state}`;
  if (!reduced && !bar.closest(".xylo.helix-on")) {
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

/* ---------- helix: playable bars → double helix → bars ----------
   One rAF loop owns every bar transform. `morph` (0 = xylophone row, 1 = helix)
   is driven by two scrubbed ScrollTriggers: the hero pin coils bars into a
   multi-turn double helix (each claim = one base-pair rung), the waitlist
   reassembly un-coils them back into a mirrored row. `spin` accumulates a
   gentle auto-rotation plus pointer/touch drag with inertia, so the helix is
   always alive and grabbable — like the OrbitControls DNA demo it's inspired
   by. Bars keep their note + verdict colours; the helix just re-uses them. */
if (!reduced) {
  const xylo = document.querySelector<HTMLElement>(".hero .xylo")!;
  const scene = document.querySelector<HTMLElement>(".hero .xylo-scene")!;
  const heroBars = xylo ? Array.from(xylo.querySelectorAll<HTMLElement>(".bar")) : [];
  if (xylo && scene && heroBars.length) {
    const n = heroBars.length;
    const TURNS = 2.5;       // full twists along the helix
    const HELIX_H = 320;     // vertical extent (px)
    const R = 120;           // strand radius / half rung length (px)
    const RUNG_LEN = R * 2;  // every claim becomes an equal-length rung
    const AUTO = 0.5;        // auto-rotation (rad/s)
    const BASE_O = 0.22;     // resting opacity while fixed in the background
    const BASE_S = 0.85;     // resting scale while fixed

    // per-bar helix geometry: bar i is a rung at height y, phase theta
    const bars = heroBars.map((el, i) => {
      const t = n > 1 ? i / (n - 1) : 0.5;
      const theta = t * Math.PI * 2 * TURNS;
      const y = (t - 0.5) * HELIX_H;
      const h = el.offsetHeight || 100;
      return { el, i, theta, y, rungScale: RUNG_LEN / h };
    });

    // natural flex centres (relative to the xylo) — the m = 0 row targets
    let rowCX: number[] = [];
    let rowCY: number[] = [];
    function captureRow() {
      const cx = xylo!.clientWidth / 2;
      const cy = xylo!.clientHeight / 2;
      rowCX = bars.map((b) => b.el.offsetLeft + b.el.offsetWidth / 2 - cx);
      rowCY = bars.map((b) => b.el.offsetTop + b.el.offsetHeight / 2 - cy);
    }
    captureRow();
    let resizeT: ReturnType<typeof setTimeout> | undefined;
    addEventListener("resize", () => {
      clearTimeout(resizeT);
      resizeT = setTimeout(captureRow, 150);
    });

    // shared state
    let morph = 0;        // 0 = row, 1 = helix
    let spin = 0;         // accumulated rotation (rad)
    let vel = 0;          // drag velocity, decays to 0 (rad/s)
    let fixed = false;    // scene detached to body as the persistent helix?
    let reconP = 0;       // 0 = free helix, 1 = reassembled at the waitlist
    let dragging = false;
    let dragMoved = false;
    let dragLastX = 0;
    let dragBar: HTMLElement | null = null;
    let running = false;
    let rafId = 0;
    let lastTs = 0;

    // Build each bar's transform from `morph` + `spin`. The bar (a vertical
    // element whose height is its rung length) is scaled to a uniform rung,
    // laid flat (rotateZ 90·m), yawed to its rung phase (rotateY m·(θ+spin)),
    // and translated from its flex spot onto the helix axis. CSS perspective
    // foreshortens the rungs into a real 3D double helix.
    function apply() {
      for (const b of bars) {
        const ang = morph * (b.theta + spin);   // rung angle (rad)
        const rz = 90 * morph;                    // lay flat as a rung
        const s = 1 + (b.rungScale - 1) * morph;  // uniform rung length
        const tx = morph * -rowCX[b.i];
        const ty = morph * (b.y - rowCY[b.i]);
        b.el.style.transform =
          `translate3d(${tx.toFixed(2)}px,${ty.toFixed(2)}px,0)` +
          ` rotateY(${(ang * 180 / Math.PI).toFixed(2)}deg)` +
          ` rotateZ(${rz.toFixed(2)}deg) scaleY(${s.toFixed(3)})`;
      }
    }

    function frame(ts: number) {
      if (!running) return;
      const dt = Math.min(0.05, (ts - lastTs) / 1000 || 0);
      lastTs = ts;
      if (!dragging) {
        spin += (AUTO + vel) * dt;
        vel *= 0.94;                 // inertia decay
        if (Math.abs(vel) < 0.002) vel = 0;
      }
      apply();
      // idle out when neither fixed, morphing, nor grabbed (saves battery)
      if ((!fixed || reconP > 0.98) && morph < 0.003 && !dragging) { running = false; return; }
      rafId = requestAnimationFrame(frame);
    }
    function ensureRunning() {
      if (!running) { running = true; lastTs = performance.now(); rafId = requestAnimationFrame(frame); }
    }
    // pause when the tab is hidden
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) { cancelAnimationFrame(rafId); running = false; }
      else if (fixed || morph > 0.003 || dragging) ensureRunning();
    });

    // drag to spin (pointer + touch). Starts on a rung, so it never blocks the
    // text scrolling behind the helix. A drag suppresses the click-note.
    function onDown(e: PointerEvent) {
      if (!fixed) return;
      dragging = true; dragMoved = false; dragLastX = e.clientX; vel = 0;
      dragBar = e.currentTarget as HTMLElement;
      ensureRunning();
    }
    function onMove(e: PointerEvent) {
      if (!dragging) return;
      const dx = e.clientX - dragLastX;
      if (Math.abs(dx) > 3) dragMoved = true;
      spin += dx * 0.01;
      vel = dx * 0.05;
      dragLastX = e.clientX;
    }
    function onUp() {
      if (dragMoved && dragBar) {
        dragBar.dataset.dragged = "1";
        setTimeout(() => { if (dragBar) delete dragBar.dataset.dragged; }, 80);
      }
      dragging = false;
    }
    addEventListener("pointermove", onMove);
    addEventListener("pointerup", onUp);
    addEventListener("pointercancel", onUp);
    for (const b of bars) b.el.addEventListener("pointerdown", onDown);

    // hint badge — echoes the DNA demo's "base pairs · double helix" readout
    const badge = document.createElement("div");
    badge.className = "helix-badge";
    badge.innerHTML = `${n} base pairs · <b>drag to spin</b>`;

    const heroEl = document.querySelector<HTMLElement>(".hero")!;
    const heroCopy = heroEl.querySelector(".hero-copy");
    function fixScene() {
      fixed = true;
      document.body.appendChild(scene);
      scene.classList.add("helix-fixed", "helix-on");
      xylo.classList.add("helix-on");
      gsap.set(scene, { xPercent: -50, yPercent: -50, rotateY: 0, scale: BASE_S, opacity: BASE_O });
      document.body.appendChild(badge);
      ensureRunning();
    }
    function unfixScene() {
      fixed = false;
      scene.classList.remove("helix-fixed", "helix-on");
      xylo.classList.remove("helix-on");
      gsap.set(scene, { clearProps: "transform,opacity" });
      gsap.set(xylo, { rotateY: 0 });
      for (const b of bars) b.el.style.transform = "";
      badge.remove();
      if (heroCopy && heroCopy.nextSibling) heroEl.insertBefore(scene, heroCopy.nextSibling);
      else heroEl.appendChild(scene);
    }

    // Phase A — pinned hero: the row coils into a double helix
    const helixTl = gsap.timeline({
      scrollTrigger: {
        trigger: ".hero",
        start: "top top",
        end: "+=220%",
        scrub: 0.6,
        pin: true,
        anticipatePin: 1,
        onEnter: () => { captureRow(); ensureRunning(); },
        onUpdate: (self) => { morph = self.progress; ensureRunning(); },
        onLeave: () => fixScene(),
        onEnterBack: () => unfixScene(),
      },
    });
    helixTl.to(".hero-copy", { yPercent: -18, opacity: 0, ease: "power2.in", duration: 0.4 }, 0);
    helixTl.to(".hero .xylo-readout", { opacity: 0, duration: 0.25 }, 0);
    helixTl.to(".hero .scroll-cue", { opacity: 0, duration: 0.2 }, 0);

    // Phase B — persistent background helix: brighten + breathe at each beat
    const pop = (peakO: number, peakS: number) => {
      if (!fixed || reconP > 0.02) return;
      gsap.to(scene, { opacity: peakO, scale: peakS, duration: 0.7, ease: "power2.out", overwrite: "auto" });
      gsap.to(scene, { opacity: BASE_O, scale: BASE_S, duration: 1.4, ease: "power1.in", delay: 0.8, overwrite: false });
    };
    document.querySelectorAll(".beat").forEach((beat) => {
      ScrollTrigger.create({ trigger: beat, start: "top 82%", onEnter: () => pop(0.6, 1.07), onEnterBack: () => pop(0.6, 1.07) });
    });
    [".trace .trace-head", ".replay .trace-head", ".artifacts h2", ".waitlist .trace-head"].forEach((sel) => {
      const el = document.querySelector(sel);
      if (!el) return;
      ScrollTrigger.create({ trigger: el, start: "top 85%", onEnter: () => pop(0.42, 1.0), onEnterBack: () => pop(0.42, 1.0) });
    });

    // Phase C — reassembly: the helix un-coils into the mirrored xylophone.
    // morph 1→0 relaxes every rung back to a bar; the scene fades as the mirror
    // fades in — a seamless handover exactly where the fixed helix sits.
    const mirrorWrap = document.querySelector<HTMLElement>(".xylo-mirror");
    if (mirrorWrap) {
      gsap.set(mirrorWrap, { autoAlpha: 0, y: 36 });
      gsap.timeline({
        scrollTrigger: {
          trigger: mirrorWrap,
          start: "top bottom",
          end: "center center",
          scrub: 0.5,
          onEnter: () => ensureRunning(),
          onUpdate: (self) => { reconP = self.progress; morph = 1 - self.progress; ensureRunning(); },
        },
      })
        .to(scene, { opacity: 0, scale: 0.72, duration: 0.5, ease: "power2.in" }, 0.1)
        .to(mirrorWrap, { autoAlpha: 1, y: 0, duration: 0.5, ease: "power2.out" }, 0.2);
    } else {
      ScrollTrigger.create({
        trigger: "footer",
        start: "top 90%",
        onEnter: () => gsap.to(scene, { opacity: 0, duration: 0.8, overwrite: "auto" }),
        onEnterBack: () => gsap.to(scene, { opacity: BASE_O, duration: 0.5, overwrite: "auto" }),
      });
    }
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
