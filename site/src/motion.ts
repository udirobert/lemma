/* Lemma landing motion engine.
   Lenis smooth scroll + GSAP ScrollTrigger: pinned hero exit, beat reveals,
   animated counters, a scroll-accumulating audit trace, and an easeReverse
   clip-path menu. Honors prefers-reduced-motion (static, no pins). */

import gsap from "gsap";
import ScrollTrigger from "gsap/ScrollTrigger";
import Lenis from "lenis";

gsap.registerPlugin(ScrollTrigger);

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
// The 3D helix journey needs desktop-sized geometry (R=120, H=320) to read
// well. On small screens the perspective collapses and the transforms break.
// Skip the helix entirely on mobile — bars stay as a playable xylophone row.
const isMobile = matchMedia("(max-width: 640px)").matches;

/* ---------- CSS-var reader with safe fallbacks ---------- */
function cssVar(name: string, fallback: number): number {
  const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (!val) return fallback;
  const num = parseFloat(val);
  return isNaN(num) ? fallback : num;
}

function loadHelixConfig() {
  return {
    turns: cssVar("--helix-turns", 2.5),
    height: cssVar("--helix-height", 320),
    radius: cssVar("--helix-radius", 120),
    autoRotate: cssVar("--helix-auto-rotate", 0.5),
    baseOpacity: cssVar("--helix-opacity", 0.22),
    baseScale: cssVar("--helix-scale", 0.85),
    finalOpacity: cssVar("--helix-final-opacity", 0.42),
    tiltDeg: cssVar("--helix-tilt", 42),
  };
}

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
let actxSuspendT: ReturnType<typeof setTimeout> | undefined;
const readout = document.getElementById("xylo-readout");

// auto-suspend the AudioContext after 10s of silence so the browser can
// release the audio graph; it resumes on the next strike. (Mobile Safari
// is especially aggressive about backgrounding audio threads.)
function scheduleActxSuspend() {
  clearTimeout(actxSuspendT);
  actxSuspendT = setTimeout(() => {
    if (actx && actx.state === "running") void actx.suspend();
  }, 10000);
}
addEventListener("pagehide", () => {
  clearTimeout(actxSuspendT);
  if (actx) void actx.close();
  actx = null;
});

function strike(bar: HTMLElement) {
  if (bar.dataset.dragged) { delete bar.dataset.dragged; return; }
  const freq = Number(bar.dataset.freq);
  const label = bar.dataset.label ?? "";
  actx = actx ?? new AudioContext();
  if (actx.state === "suspended") void actx.resume();
  scheduleActxSuspend();
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
    bar.closest(".xylo-scene")?.querySelector<HTMLElement>(".xylo-readout") ?? readout;
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

/* ---------- helix: playable bars → double helix → verdict spectrum ----------
   One rAF loop owns every bar transform. The scroll journey is a single
   narrative arc with no dead handoff:
     Phase A — pinned hero: the row coils into a multi-turn double helix
              (each claim = one base-pair rung). `morph` 0→1.
     Phase B — fixed background helix: auto-rotation + drag-to-spin + beat
              brightness pops.
     Phase B2 — the helix leans back (rotateX hump) for a depth reveal.
     Phase B3 — the helix unwinds (`unfold` 0→1) into a flat spectrum where the
              bars re-sort by verdict (supported → inconclusive → falsified).
     Phase C — the closing bookend: colour ramps back to full, the verdict
              sort dissolves into a mirrored version of the hero row, and the
              scene parks below the footer as the page's last frame.
   `spin` accumulates gentle auto-rotation + pointer/touch drag with inertia
   (and arrow-key nudges), and a wake/idle doze keeps the loop asleep between
   interactions. Bars keep their note + verdict colours throughout. The old
   static mirror handoff is gone — the bars are always on-screen, transforming
   to the end. All helix geometry is CSS-var tunable (see loadHelixConfig).
   Skipped on mobile (< 640px) where the 3D perspective collapses — the
   bars stay as a playable xylophone row instead. */
if (!reduced && !isMobile) {
  const xylo = document.querySelector<HTMLElement>(".hero .xylo")!;
  const scene = document.querySelector<HTMLElement>(".hero .xylo-scene")!;
  const heroBars = xylo ? Array.from(xylo.querySelectorAll<HTMLElement>(".bar")) : [];
  if (xylo && scene && heroBars.length) {
    const n = heroBars.length;
    let cfg = loadHelixConfig();
    let TURNS = cfg.turns;         // full twists along the helix
    let HELIX_H = cfg.height;      // vertical extent (px)
    let R = cfg.radius;            // strand radius / half rung length (px)
    let RUNG_LEN = R * 2;          // every claim becomes an equal-length rung
    const AUTO = cfg.autoRotate;     // auto-rotation (rad/s)
    let BASE_O = cfg.baseOpacity;  // resting opacity while fixed in the background
    let BASE_S = cfg.baseScale;    // resting scale while fixed
    const FINAL_O = cfg.finalOpacity; // mid-journey spectrum brightness, before the full-colour bookend
    let TILT_DEG = cfg.tiltDeg;    // peak helix lean for the depth reveal
    const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

    // per-bar helix geometry + verdict. bar i is a rung at height y, phase
    // theta; `state` drives the spectrum sort, `rank`/`specX` its sorted slot.
    const order: Record<string, number> = { on: 0, dim: 1, fail: 2 };
    const bars = heroBars.map((el, i) => {
      const t = n > 1 ? i / (n - 1) : 0.5;
      const theta = t * Math.PI * 2 * TURNS;
      const y = (t - 0.5) * HELIX_H;
      const h = el.offsetHeight || 100;
      return {
        el, i, theta, y, rungScale: RUNG_LEN / h,
        state: el.dataset.state ?? "dim", rank: i, specX: 0,
      };
    });

    // re-read CSS vars (breakpoint may have changed on resize) and
    // recompute each bar's helix geometry to match the new values.
    function reloadHelix() {
      cfg = loadHelixConfig();
      TURNS = cfg.turns;
      HELIX_H = cfg.height;
      R = cfg.radius;
      RUNG_LEN = R * 2;
      BASE_O = cfg.baseOpacity;
      BASE_S = cfg.baseScale;
      TILT_DEG = cfg.tiltDeg;
      for (const b of bars) {
        const t = n > 1 ? b.i / (n - 1) : 0.5;
        b.theta = t * Math.PI * 2 * TURNS;
        b.y = (t - 0.5) * HELIX_H;
        b.rungScale = RUNG_LEN / (b.el.offsetHeight || 100);
      }
    }

    // natural flex centres (relative to the xylo) — the m = 0 row targets,
    // and the spectrum reuses these evenly-spaced slots in verdict-sorted order.
    let rowCX: number[] = [];
    let rowCY: number[] = [];
    function captureRow() {
      const cx = xylo!.clientWidth / 2;
      const cy = xylo!.clientHeight / 2;
      rowCX = bars.map((b) => b.el.offsetLeft + b.el.offsetWidth / 2 - cx);
      rowCY = bars.map((b) => b.el.offsetTop + b.el.offsetHeight / 2 - cy);
      // sort by verdict (supported → inconclusive → falsified), stable within
      // each verdict, then assign each bar the horizontal slot of its rank.
      const sorted = [...bars].sort(
        (a, b) => (order[a.state] ?? 1) - (order[b.state] ?? 1) || a.i - b.i
      );
      sorted.forEach((b, rank) => { b.rank = rank; b.specX = rowCX[rank]; });
    }
    captureRow();
    let resizeT: ReturnType<typeof setTimeout> | undefined;
    addEventListener("resize", () => {
      clearTimeout(resizeT);
      resizeT = setTimeout(() => { reloadHelix(); captureRow(); }, 150);
    });

    // shared state
    let morph = 0;        // 0 = row, 1 = helix (Phase A)
    let spin = 0;         // accumulated rotation (rad)
    let vel = 0;          // drag velocity, decays to 0 (rad/s)
    let fixed = false;    // scene detached to body as the persistent helix?
    let J = 0;            // post-hero journey progress (0..1): tilt + spectrum
    let unfold = 0;       // 0 = helix, 1 = flat verdict-sorted spectrum (from J)
    let tilt = 0;         // helix lean 0..1, a mid-journey hump for depth (from J)
    let dragging = false;
    let dragMoved = false;
    let dragLastX = 0;
    let dragLastT = 0;
    let dragBar: HTMLElement | null = null;
    let pendingDrag: { id: number; x: number; y: number } | null = null; // down not yet classified
    let running = false;
    let rafId = 0;
    let lastTs = 0;
    // deep-read saver: the free-running loop dozes off this many ms after the
    // last beat pop / interaction, and wakes on pops, drags, tab focus
    let awakeUntil = 0;
    const wake = (ms: number) => { awakeUntil = performance.now() + ms; ensureRunning(); };

    // Build each bar's transform from `morph` (row→helix) then `unfold`
    // (helix→verdict spectrum), plus a container `tilt` for the depth reveal.
    // The bar (a vertical element whose height is its rung length) is scaled to
    // a uniform rung, laid flat (rotateZ 90·m), yawed to its rung phase
    // (rotateY m·(θ+spin)), and translated from its flex spot onto the helix
    // axis. As `unfold` rises the rungs un-flatten, un-yaw, and slide into
    // verdict-sorted horizontal slots — the helix resolving into its answer.
    // The final act reverses the resolution: the sort dissolves into a
    // mirrored version of the hero row (the journey's bookend), the scene
    // returns to full colour, and it parks below the footer text so the
    // closing frame sits beneath all the page content.
    function apply() {
      // journey-derived scalars (only meaningful once the scene is fixed).
      // every window completes at J = 1 — the absolute page bottom — so the
      // finale is the last thing, not a mid-page afterthought.
      const j = fixed ? J : 0;
      unfold = clamp01((j - 0.62) / 0.28);                                  // 0.62→0.90 verdict sort
      tilt = clamp01((j - 0.45) / 0.12) * (1 - clamp01((j - 0.62) / 0.12)); // 0.45→0.62 depth hump
      const rev = clamp01((j - 0.9) / 0.1);                                 // 0.90→1.00 mirror
      const park = clamp01((j - 0.93) / 0.07);                              // 0.93→1.00 park low
      if (fixed) {
        xylo!.style.transform = `rotateX(${(tilt * TILT_DEG).toFixed(2)}deg)`;
        // colour: dim background helix → finale opacity → full colour bookend
        const ramp = clamp01((j - 0.6) / 0.25);
        const op = lerp(lerp(BASE_O, FINAL_O, ramp), 1, clamp01((j - 0.85) / 0.15));
        const sc = lerp(BASE_S, 1, ramp);
        gsap.set(scene, { opacity: op, scale: sc, y: park * parkY() });
      }
      // fade the verdict hairline + axis in during the unwind, and fade them
      // back out during the mirror — by then the verdict story has been told
      if (fixed && hairline) {
        const op = clamp01((j - 0.55) / 0.15) * (1 - rev);
        hairline.style.opacity = op.toFixed(2);
        axisCaption.style.opacity = op.toFixed(2);
      }

      for (const b of bars) {
        // coil: row (m=0) → helix (m=1)
        const coilTX = -rowCX[b.i] * morph;
        const coilTY = (b.y - rowCY[b.i]) * morph;
        const coilRY = (b.theta + spin) * morph;
        const coilRZ = 90 * morph;
        const coilS = 1 + (b.rungScale - 1) * morph;
        // spectrum target: natural row, but in verdict-sorted horizontal slots
        const specTX = b.specX - rowCX[b.i];
        // finale target: mirror of the hero row — bar i takes slot n-1-i
        const revTX = rowCX[n - 1 - b.i] - rowCX[b.i];
        // compose: coil → verdict spectrum (unfold) → mirrored row (rev)
        const flat = Math.max(unfold, rev);
        const tx = lerp(lerp(coilTX, specTX, unfold), revTX, rev);
        const ty = lerp(coilTY, 0, flat);
        const ry = lerp(coilRY, 0, flat);
        const rz = lerp(coilRZ, 0, flat);
        const s = lerp(coilS, 1, flat);
        b.el.style.transform =
          `translate3d(${tx.toFixed(2)}px,${ty.toFixed(2)}px,0)` +
          ` rotateY(${(ry * 180 / Math.PI).toFixed(2)}deg)` +
          ` rotateZ(${rz.toFixed(2)}deg) scaleY(${s.toFixed(3)})`;
      }
    }

    // vertical offset that parks the scene just above the viewport bottom,
    // so at the page end the bars sit below the footer text
    let sceneH = 0;
    function parkY() {
      return innerHeight / 2 - sceneH / 2 - 28;
    }

    function frame(ts: number) {
      if (!running) return;
      const dt = lastTs ? Math.min(0.05, (ts - lastTs) / 1000) : 0;
      lastTs = ts;
      // spin ownership: once the spectrum begins to unfold (J past 0.55) the
      // journey owns the geometry; auto-spin/velocity are still allowed to
      // decay but no longer drive rotation so the un-coil reads as deliberate.
      if (!dragging && unfold <= 0) {
        spin += (AUTO + vel) * dt;
        vel *= Math.exp(-3.7 * dt);    // frame-rate-independent inertia decay
        if (Math.abs(vel) < 0.002) vel = 0;
      } else if (!dragging) {
        vel *= Math.exp(-3.7 * dt);
        if (Math.abs(vel) < 0.002) vel = 0;
      }
      apply();
      // sleep when nothing needs us: the row is settled, or the fixed helix
      // is dozing between pops/interactions. While fixed, all geometry is a
      // pure function of J (scroll-driven — every J update wakes us), so
      // dozing mid-journey is always safe. awakeUntil is refreshed by pops,
      // drags, keyboard, and the journey's scroll onUpdate.
      const idleRow = !fixed && morph < 0.003;
      const dozing = performance.now() >= awakeUntil && Math.abs(vel) < 0.002;
      if (!dragging && (idleRow || (fixed && dozing))) { running = false; return; }
      rafId = requestAnimationFrame(frame);
    }
    function ensureRunning() {
      if (!running) { running = true; lastTs = performance.now(); rafId = requestAnimationFrame(frame); }
    }
    // pause when the tab is hidden; on return, the helix gets a short grace
    // window before it dozes again
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) { cancelAnimationFrame(rafId); running = false; }
      else if (fixed || morph > 0.003 || dragging) {
        if (fixed) wake(2500);
        else ensureRunning();
      }
    });

    // wake when scrolling back up through the page (the reader is heading
    // toward content again); downward reading stays asleep
    let lastY = scrollY;
    addEventListener(
      "scroll",
      () => {
        const y = scrollY;
        if (y < lastY && fixed && performance.now() >= awakeUntil) wake(3000);
        lastY = y;
      },
      { passive: true }
    );

    // drag to spin (pointer + touch). Starts on a rung, so it never blocks the
    // text scrolling behind the helix. Intent is classified once: a
    // predominantly horizontal gesture grabs the helix, a vertical one falls
    // through to native scrolling (touch-action: pan-y). A real drag
    // suppresses the click-note; a tap still plays.
    function onDown(e: PointerEvent) {
      if (!fixed) return;
      if (e.pointerType === "mouse" && e.button !== 0) return;
      pendingDrag = { id: e.pointerId, x: e.clientX, y: e.clientY };
      dragMoved = false; vel = 0;
      dragBar = e.currentTarget as HTMLElement;
      wake(6000);
    }
    function onMove(e: PointerEvent) {
      if (pendingDrag) {
        if (e.pointerId !== pendingDrag.id) return;
        const dx = e.clientX - pendingDrag.x;
        const dy = e.clientY - pendingDrag.y;
        if (Math.abs(dx) < 6 && Math.abs(dy) < 6) return; // dead zone: still deciding
        pendingDrag = null;
        if (Math.abs(dx) < Math.abs(dy)) return;          // vertical intent → let the page scroll
        dragging = true;
        dragLastX = e.clientX;
        dragLastT = e.timeStamp;
        try { dragBar?.setPointerCapture(e.pointerId); } catch { /* stale id */ }
        ensureRunning();
        return;
      }
      if (!dragging || !dragBar?.hasPointerCapture?.(e.pointerId)) return;
      const dx = e.clientX - dragLastX;
      const dms = Math.max(8, e.timeStamp - dragLastT);
      dragLastT = e.timeStamp;
      if (Math.abs(dx) > 3) dragMoved = true;
      spin += dx * 0.01;
      vel = Math.max(-10, Math.min(10, dx * 0.01 / (dms / 1000))); // device-independent release speed
      dragLastX = e.clientX;
    }
    function onUp(e?: PointerEvent) {
      if (!dragging && !pendingDrag) return;
      pendingDrag = null;
      if (e && dragging && dragBar?.hasPointerCapture?.(e.pointerId)) {
        try { dragBar.releasePointerCapture(e.pointerId); } catch { /* already released */ }
      }
      if (dragMoved && dragBar) {
        dragBar.dataset.dragged = "1";
        setTimeout(() => { if (dragBar) delete dragBar.dataset.dragged; }, 80);
      }
      dragging = false;
      dragMoved = false;
    }
    addEventListener("pointermove", onMove);
    addEventListener("pointerup", onUp as EventListener, true);
    addEventListener("pointercancel", onUp as EventListener, true);
    for (const b of bars) b.el.addEventListener("pointerdown", onDown);

    // keyboard controls for the fixed helix — arrow keys spin, Enter/Space strikes a note
    function onKey(e: KeyboardEvent) {
      if (!fixed) return;
      const key = e.key;
      if (key === "ArrowLeft") {
        e.preventDefault();
        spin -= 0.15;
        vel = -0.15;
        wake(4000);
      } else if (key === "ArrowRight") {
        e.preventDefault();
        spin += 0.15;
        vel = 0.15;
        wake(4000);
      } else if ((key === "Enter" || key === " ") && document.activeElement) {
        e.preventDefault();
        const el = document.activeElement as HTMLElement | null;
        if (el?.classList.contains("bar")) strike(el);
      }
    }
    addEventListener("keydown", onKey);

    // hint badge — echoes the DNA demo's "base pairs · double helix" readout
    const badge = document.createElement("div");
    badge.className = "helix-badge";
    badge.innerHTML = `${n} base pairs · <b>drag to spin</b>`;

    // spectrum hairline — gradient bar at the bottom that fades in during the unwind
    const hairline = document.createElement("div");
    hairline.className = "spectrum-fixed";
    hairline.style.opacity = "0";

    // spectrum axis caption — labels the verdict sort so the finale reads as
    // an answer, not just a rearrangement. Fades in with the hairline.
    const axisCaption = document.createElement("div");
    axisCaption.className = "spectrum-axis";
    axisCaption.innerHTML =
      '<span class="ax-l">supported</span>' +
      '<span class="ax-m">inconclusive</span>' +
      '<span class="ax-r">falsified</span>';
    axisCaption.style.opacity = "0";

    const heroEl = document.querySelector<HTMLElement>(".hero")!;
    const heroCopy = heroEl.querySelector(".hero-copy");
    function fixScene() {
      fixed = true;
      document.body.appendChild(scene);
      scene.classList.add("helix-fixed", "helix-on");
      xylo.classList.add("helix-on");
      gsap.set(scene, { xPercent: -50, yPercent: -50, rotateY: 0, scale: BASE_S, opacity: BASE_O });
      sceneH = scene.offsetHeight; // cached for the finale park offset
      document.body.appendChild(badge);
      document.body.appendChild(hairline);
      document.body.appendChild(axisCaption);
      wake(6000);
      // make bars focusable for keyboard accessibility
      for (const b of bars) {
        b.el.setAttribute("tabindex", "0");
      }
    }
    function unfixScene() {
      fixed = false;
      J = 0; unfold = 0; tilt = 0;
      scene.classList.remove("helix-fixed", "helix-on");
      xylo.classList.remove("helix-on");
      gsap.set(scene, { clearProps: "transform,opacity,filter" });
      gsap.set(xylo, { clearProps: "transform" });
      // the Phase A timeline animates hero-copy to yPercent:-18 / opacity:0;
      // clear it so the re-inserted scene lines up with a non-transformed hero.
      if (heroCopy) gsap.set(heroCopy, { clearProps: "transform,opacity" });
      for (const b of bars) b.el.style.transform = "";
      badge.remove();
      hairline.remove();
      axisCaption.remove();
      // clean up keyboard listeners and tabindex
      for (const b of bars) {
        b.el.removeAttribute("tabindex");
      }
      // hard-reset accumulated rotation/journey so re-entering the hero
      // never replays leftover spin (fix 3: state reset)
      morph = 0; spin = 0; vel = 0;
      pendingDrag = null; dragging = false; dragMoved = false;
      cancelAnimationFrame(rafId); running = false;
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

    // Phase B — persistent background helix: a brightness pulse at each beat.
    // Pops use `filter: brightness` (not opacity/scale) so the journey
    // timeline can own opacity/scale for the finale without any conflict.
    const pop = (peakB: number) => {
      if (!fixed) return;              // only pulse the fixed helix
      wake(4500);
      gsap.to(scene, { filter: `brightness(${peakB})`, duration: 0.7, ease: "power2.out", overwrite: "auto" });
      gsap.to(scene, { filter: "brightness(1)", duration: 1.4, ease: "power1.in", delay: 0.8, overwrite: false });
    };
    document.querySelectorAll(".beat").forEach((beat) => {
      ScrollTrigger.create({ trigger: beat, start: "top 82%", onEnter: () => pop(1.8), onEnterBack: () => pop(1.8) });
    });
    [".trace .trace-head", ".replay .trace-head", ".artifacts h2", ".waitlist .trace-head"].forEach((sel) => {
      const el = document.querySelector(sel);
      if (!el) return;
      ScrollTrigger.create({ trigger: el, start: "top 85%", onEnter: () => pop(1.45), onEnterBack: () => pop(1.45) });
    });

    // Phase B2 + B3 + C — the post-hero journey: one scrubbed ScrollTrigger
    // from the first story beat to the absolute page bottom. Its progress `J`
    // drives everything per-frame in apply(): the depth-lean hump, the unfold
    // into the verdict spectrum, the colour ramp back to full colour, the
    // mirror into reversed hero order, and the final park below the footer.
    // No tween owns scene opacity/scale anymore — apply() does, so the whole
    // closing act is a pure function of scroll position. No mirror handoff —
    // the bars are always on-screen, transforming all the way down.
    ScrollTrigger.create({
      trigger: ".story",
      start: "top 85%",
      endTrigger: "footer",
      end: "bottom bottom",
      onUpdate: (self) => { J = self.progress; wake(2500); },
    });
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
      // pinnedContainer requires GSAP 3.11+ (ScrollTrigger). If the project
      // ever downgrades GSAP, this option is silently ignored and the pin
      // may double its spacer — verify the pinned GSAP version in package.json.
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
