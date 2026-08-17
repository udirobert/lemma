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

/* ---------- pinned hero: row -> DNA helix -> persistent spin ---------- */
if (!reduced) {
  const xylo = document.querySelector<HTMLElement>(".hero .xylo");
  const scene = document.querySelector<HTMLElement>(".hero .xylo-scene");
  const heroBars = xylo ? Array.from(xylo.querySelectorAll<HTMLElement>(".bar")) : [];
  if (xylo && scene && heroBars.length) {
    const n = heroBars.length;
    const R = 150; // helix ring radius in px
    const spacing = 30; // vertical gap between rungs
    const cx = xylo.clientWidth / 2;
    const cy = xylo.clientHeight / 2;

    // Helix targets: each bar placed on a ring via sin/cos, laid horizontal as a
    // rung. x/y are offsets from the bar's natural flex position so the ring
    // ends up centered in the xylophone container.
    const targets = heroBars.map((bar, i) => {
      const angle = (i / n) * Math.PI * 2; // radians
      const natX = bar.offsetLeft + bar.offsetWidth / 2;
      const natY = bar.offsetTop + bar.offsetHeight / 2;
      return {
        x: cx - natX + R * Math.sin(angle),
        y: cy - natY + (i - (n - 1) / 2) * spacing,
        z: R * Math.cos(angle),
        rotY: (i / n) * 360,
      };
    });

    // Phase A: pinned assembly + spin
    let handoffScroll = 0;
    const heroEl = document.querySelector<HTMLElement>(".hero");
    const heroCopy = heroEl!.querySelector(".hero-copy");

    const helixTl = gsap.timeline({
      scrollTrigger: {
        trigger: ".hero",
        start: "top top",
        end: "+=250%",
        scrub: 0.6,
        pin: true,
        anticipatePin: 1,
        onLeave: () => {
          handoffScroll = window.scrollY;
          document.body.appendChild(scene);
          scene.classList.add("helix-fixed");
          gsap.set(scene, {
            xPercent: -50,
            yPercent: -50,
            rotateY: 0,
            scale: 0.85,
            opacity: 0.16,
          });
        },
        onEnterBack: () => {
          scene.classList.remove("helix-fixed");
          gsap.set(scene, { clearProps: "transform,opacity" });
          gsap.set(xylo, { rotateY: 0 });
          if (heroCopy && heroCopy.nextSibling) {
            heroEl!.insertBefore(scene, heroCopy.nextSibling);
          } else {
            heroEl!.appendChild(scene);
          }
        },
      },
    });

    // clear the stage
    helixTl.to(".hero-copy", { yPercent: -18, opacity: 0, ease: "power2.in", duration: 0.4 }, 0);
    helixTl.to(".hero .xylo-readout", { opacity: 0, duration: 0.25 }, 0);
    helixTl.to(".hero .scroll-cue", { opacity: 0, duration: 0.2 }, 0);

    // bars fly into helix ring
    heroBars.forEach((bar, i) => {
      helixTl.to(
        bar,
        {
          x: targets[i].x,
          y: targets[i].y,
          z: targets[i].z,
          rotateY: targets[i].rotY,
          rotationZ: 90,
          transformOrigin: "50% 50%",
          ease: "power2.inOut",
          duration: 1.0,
        },
        0.3 + i * 0.035
      );
    });

    // spin the whole helix one full turn while pinned (container rotation,
    // preserve-3d keeps bars on the ring)
    helixTl.to(xylo, { rotateY: 360, ease: "none", duration: 1.2 }, 1.6);
  }

  /* ---------- persistent helix: master rotation + foreground pops ---------- */
  const fixedScene = document.querySelector<HTMLElement>(".xylo-scene");
  const fixedXylo = document.querySelector<HTMLElement>(".xylo-scene .xylo");
  if (fixedScene && fixedXylo) {
    let handoffScroll = 0;
    const heroEl = document.querySelector<HTMLElement>(".hero");

    // Continuous rotation of the helix container (perspective stays on scene),
    // mapped so it starts at 0 exactly where the pin released.
    ScrollTrigger.create({
      trigger: document.body,
      start: "top top",
      end: "bottom bottom",
      onUpdate: () => {
        if (!fixedScene.classList.contains("helix-fixed")) return;
        const total = document.documentElement.scrollHeight - innerHeight;
        const denom = Math.max(1, total - handoffScroll);
        const since = Math.max(0, (window.scrollY - handoffScroll) / denom);
        gsap.set(fixedXylo, { rotateY: since * 540 });
      },
    });

    // Foreground pop at each story beat: helix brightens + scales, then recedes.
    const pop = (peakOpacity: number, peakScale: number) => {
      if (!fixedScene.classList.contains("helix-fixed")) return;
      gsap.to(fixedScene, { opacity: peakOpacity, scale: peakScale, duration: 0.7, ease: "power2.out", overwrite: "auto" });
      gsap.to(fixedScene, { opacity: 0.14, scale: 0.8, duration: 1.5, ease: "power1.in", delay: 0.9, overwrite: false });
    };
    document.querySelectorAll(".beat").forEach((beat) => {
      ScrollTrigger.create({
        trigger: beat,
        start: "top 82%",
        onEnter: () => pop(0.55, 1.06),
        onEnterBack: () => pop(0.55, 1.06),
      });
    });

    // Softer pop at the remaining section heads.
    [".trace .trace-head", ".replay .trace-head", ".artifacts h2", ".waitlist .trace-head"].forEach((sel) => {
      const el = document.querySelector(sel);
      if (!el) return;
      ScrollTrigger.create({
        trigger: el,
        start: "top 85%",
        onEnter: () => pop(0.4, 1.0),
        onEnterBack: () => pop(0.4, 1.0),
      });
    });

    // Final bow: helix fades out entirely at the footer.
    ScrollTrigger.create({
      trigger: "footer",
      start: "top 90%",
      onEnter: () => gsap.to(fixedScene, { opacity: 0, duration: 0.8, overwrite: "auto" }),
      onEnterBack: () => gsap.to(fixedScene, { opacity: 0.14, duration: 0.5, overwrite: "auto" }),
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
