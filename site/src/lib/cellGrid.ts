/* Spatial-grid pair separation for the organic-cells backdrop.
   Extracted from BgGlass.astro so the collision logic is unit-testable
   without a WebGL context. Mutates cell velocities in place, same as the
   inline version did. */

export type Cell = { x: number; y: number; vx: number; vy: number; r: number };

export type SeparationConfig = {
  /** max cell radius — sets the grid bucket size (2× this) */
  maxRadius: number;
  /** separation force px/s² applied per overlapping pair */
  pairPush: number;
  /** collision distance multiplier (cells separate at (r1+r2)*0.92) */
  contactFactor: number;
};

/**
 * Apply pairwise separation forces using a uniform spatial grid.
 *
 * For each pair of cells closer than `(a.r + b.r) * contactFactor`, apply
 * an equal-and-opposite push along the contact normal, scaled by overlap
 * depth. The grid makes this O(n) in the average case instead of O(n²).
 *
 * Mutates `cells[*].vx` / `cells[*].vy` in place.
 */
export function applySeparation(
  cells: Cell[],
  w: number,
  h: number,
  cfg: SeparationConfig,
  dt: number,
): void {
  const cellSize = cfg.maxRadius * 2;
  const gx = Math.max(1, Math.ceil(w / cellSize));
  const gy = Math.max(1, Math.ceil(h / cellSize));
  const grid: number[][] = new Array(gx * gy);

  // bin each cell into its bucket
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    const cx = Math.min(gx - 1, Math.max(0, Math.floor(c.x / cellSize)));
    const cy = Math.min(gy - 1, Math.max(0, Math.floor(c.y / cellSize)));
    const k = cy * gx + cx;
    (grid[k] ??= []).push(i);
  }

  // check each bucket + its 4 forward neighbors (right, down-left, down, down-right).
  // The backward neighbors are covered when the backward bucket iterates forward,
  // so we don't need a checked-set — the `i < j` dedupe handles intra-bucket pairs.
  const neighborOffsets: [number, number][] = [
    [0, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
  ];

  for (let cy = 0; cy < gy; cy++) {
    for (let cx = 0; cx < gx; cx++) {
      const bucket = grid[cy * gx + cx];
      if (!bucket) continue;

      // For the same-bucket offset [0,0], dedupe with i < j to avoid checking
      // each intra-bucket pair twice. For cross-bucket offsets, the pair
      // (bucket, other) is only visited from this direction (the backward
      // direction is covered when the other bucket iterates forward), so
      // check ALL pairs between the two buckets without an i < j guard.
      for (const [dx, dy] of neighborOffsets) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || nx >= gx || ny < 0 || ny >= gy) continue;
        const other = grid[ny * gx + nx];
        if (!other) continue;
        const isSame = dx === 0 && dy === 0;

        for (const i of bucket) {
          for (const j of other) {
            if (isSame && i >= j) continue; // dedupe intra-bucket pairs only
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
    }
  }
}
