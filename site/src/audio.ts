let actx: AudioContext | null = null;
let actxSuspendT: ReturnType<typeof setTimeout> | undefined;

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

export function playStrike(freq: number) {
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
}
