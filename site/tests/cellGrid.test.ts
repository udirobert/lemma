import { describe, it, expect } from "vitest";
import { applySeparation, type Cell } from "../src/lib/cellGrid";

/* Helpers -------------------------------------------------------------- */

function makeCell(
  x: number,
  y: number,
  r: number,
  vx = 0,
  vy = 0,
): Cell {
  return { x, y, vx, vy, r };
}

const CFG = {
  maxRadius: 34,
  pairPush: 70,
  contactFactor: 0.92,
};
const DT = 0.016; // ~60fps frame
const VIEW_W = 1920;
const VIEW_H = 1080;

/* Brute-force reference — O(n²), the ground truth the grid must match. */
function bruteSeparation(cells: Cell[], w: number, h: number, cfg: typeof CFG, dt: number) {
  for (let i = 0; i < cells.length; i++) {
    for (let j = i + 1; j < cells.length; j++) {
      const a = cells[i];
      const b = cells[j];
      const ddx = b.x - a.x;
      const ddy = b.y - a.y;
      const minD = (a.r + b.r) * cfg.contactFactor;
      const d2 = ddx * ddx + ddy * ddy;
      if (d2 > 0.01 && d2 < minD * minD) {
        const d = Math.sqrt(d2);
        const push = ((minD - d) / minD) * cfg.pairPush * dt;
        const ux = ddx / d;
        const uy = ddy / d;
        a.vx -= ux * push;
        a.vy -= uy * push;
        b.vx += ux * push;
        b.vy += uy * push;
      }
    }
  }
}

function clone(cells: Cell[]): Cell[] {
  return cells.map((c) => ({ ...c }));
}

/* Tests ---------------------------------------------------------------- */

describe("applySeparation", () => {
  it("does nothing when cells are far apart", () => {
    const cells = [
      makeCell(100, 100, 20),
      makeCell(500, 500, 20),
      makeCell(1000, 800, 20),
    ];
    const before = clone(cells);
    applySeparation(cells, VIEW_W, VIEW_H, CFG, DT);
    expect(cells).toEqual(before);
  });

  it("pushes overlapping cells apart along the contact normal", () => {
    // two cells touching — separation should push them directly apart on x
    const cells = [makeCell(200, 500, 30), makeCell(220, 500, 30)];
    applySeparation(cells, VIEW_W, VIEW_H, CFG, DT);

    // cell 0 pushed left (−x), cell 1 pushed right (+x)
    expect(cells[0].vx).toBeLessThan(0);
    expect(cells[1].vx).toBeGreaterThan(0);
    // no y velocity since they're aligned on y
    expect(cells[0].vy).toBeCloseTo(0, 10);
    expect(cells[1].vy).toBeCloseTo(0, 10);
    // equal and opposite (Newton's 3rd law)
    expect(cells[0].vx).toBeCloseTo(-cells[1].vx, 10);
  });

  it("pushes harder when cells overlap more (deeper penetration)", () => {
    const shallow = [makeCell(200, 500, 30), makeCell(212, 500, 30)];
    const deep = [makeCell(200, 500, 30), makeCell(202, 500, 30)];
    applySeparation(shallow, VIEW_W, VIEW_H, CFG, DT);
    applySeparation(deep, VIEW_W, VIEW_H, CFG, DT);
    // deeper overlap → larger push magnitude
    expect(Math.abs(deep[0].vx)).toBeGreaterThan(Math.abs(shallow[0].vx));
  });

  it("produces zero net momentum change (conservation)", () => {
    const cells = [
      makeCell(200, 500, 25),
      makeCell(230, 510, 30),
      makeCell(210, 540, 20),
    ];
    applySeparation(cells, VIEW_W, VIEW_H, CFG, DT);
    const totalVx = cells.reduce((s, c) => s + c.vx, 0);
    const totalVy = cells.reduce((s, c) => s + c.vy, 0);
    expect(totalVx).toBeCloseTo(0, 10);
    expect(totalVy).toBeCloseTo(0, 10);
  });

  it("matches brute-force O(n²) result on random layouts", () => {
    const rng = mulberry32(42); // deterministic seed
    for (let trial = 0; trial < 20; trial++) {
      const n = 3 + Math.floor(rng() * 12); // 3–14 cells
      const cells: Cell[] = [];
      for (let i = 0; i < n; i++) {
        cells.push(
          makeCell(
            rng() * 400,
            rng() * 400,
            12 + rng() * 22,
          ),
        );
      }
      const gridVersion = clone(cells);
      const bruteVersion = clone(cells);
      applySeparation(gridVersion, 800, 800, CFG, DT);
      bruteSeparation(bruteVersion, 800, 800, CFG, DT);
      for (let i = 0; i < n; i++) {
        expect(gridVersion[i].vx).toBeCloseTo(bruteVersion[i].vx, 8);
        expect(gridVersion[i].vy).toBeCloseTo(bruteVersion[i].vy, 8);
      }
    }
  });

  it("matches brute-force on a dense cluster (stress test)", () => {
    // all cells packed into a tight area — max grid bucket collisions
    const rng = mulberry32(99);
    const cells: Cell[] = [];
    for (let i = 0; i < 30; i++) {
      cells.push(makeCell(rng() * 60, rng() * 60, 12 + rng() * 22));
    }
    const gridVersion = clone(cells);
    const bruteVersion = clone(cells);
    applySeparation(gridVersion, 800, 800, CFG, DT);
    bruteSeparation(bruteVersion, 800, 800, CFG, DT);
    for (let i = 0; i < cells.length; i++) {
      expect(gridVersion[i].vx).toBeCloseTo(bruteVersion[i].vx, 8);
      expect(gridVersion[i].vy).toBeCloseTo(bruteVersion[i].vy, 8);
    }
  });

  it("handles cells at exact same position (no NaN)", () => {
    const cells = [makeCell(200, 500, 30), makeCell(200, 500, 30)];
    applySeparation(cells, VIEW_W, VIEW_H, CFG, DT);
    // d2 ≤ 0.01 guard skips the pair → velocities stay 0, no NaN
    for (const c of cells) {
      expect(Number.isFinite(c.vx)).toBe(true);
      expect(Number.isFinite(c.vy)).toBe(true);
    }
  });

  it("handles empty cell array without error", () => {
    const cells: Cell[] = [];
    expect(() => applySeparation(cells, VIEW_W, VIEW_H, CFG, DT)).not.toThrow();
  });

  it("handles a single cell (no pairs to check)", () => {
    const cells = [makeCell(200, 500, 30)];
    const before = clone(cells);
    applySeparation(cells, VIEW_W, VIEW_H, CFG, DT);
    expect(cells).toEqual(before);
  });

  it("handles viewport smaller than one grid cell", () => {
    // tiny viewport → gx=1, gy=1 → single bucket, all pairs checked intra-bucket
    const cells = [makeCell(10, 10, 20), makeCell(15, 15, 20)];
    applySeparation(cells, 30, 30, CFG, DT);
    expect(cells[0].vx).not.toBeCloseTo(0, 5);
    expect(cells[1].vx).not.toBeCloseTo(0, 5);
  });
});

/* deterministic PRNG for reproducible test layouts ---------------------- */
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
