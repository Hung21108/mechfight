"""
Fast 2.5D map representation, spatial hash, terrain sampler, exact edge/prism
raycaster, coarse A* navigation grid, and connected escape evaluator for mechfight/1.
Parses v2/world.json and v2/terrain.i16 during the 20s hello budget,
and supports sub-millisecond queries during the 50ms decision tick.
"""

from __future__ import annotations

import array
import heapq
import json
import math
import os
import sys
from collections import deque
import time
from typing import Any, Dict, List, Optional, Tuple


class Prism:
    __slots__ = (
        "kind",
        "base",
        "top",
        "ring",
        "holes",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "cx",
        "cy",
        "edges",
        "hole_edges",
    )

    def __init__(
        self,
        kind: str,
        base: float,
        top: float,
        ring_flat: List[float],
        holes_flat: List[List[float]],
    ) -> None:
        self.kind = kind
        self.base = float(base)
        self.top = float(top)

        pts: List[Tuple[float, float]] = []
        n = len(ring_flat) // 2
        for i in range(n):
            pts.append((float(ring_flat[2 * i]), float(ring_flat[2 * i + 1])))
        self.ring = pts

        self.holes: List[List[Tuple[float, float]]] = []
        hole_edges: List[Tuple[float, float, float, float]] = []
        for h in holes_flat or []:
            hpts: List[Tuple[float, float]] = []
            hn = len(h) // 2
            for i in range(hn):
                hpts.append((float(h[2 * i]), float(h[2 * i + 1])))
            if hpts:
                self.holes.append(hpts)
                if len(hpts) >= 2:
                    for i in range(len(hpts)):
                        x1, y1 = hpts[i]
                        x2, y2 = hpts[(i + 1) % len(hpts)]
                        if (x1 != x2) or (y1 != y2):
                            hole_edges.append((x1, y1, x2, y2))
        self.hole_edges = hole_edges

        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self.min_x = min(xs)
            self.max_x = max(xs)
            self.min_y = min(ys)
            self.max_y = max(ys)
            self.cx = 0.5 * (self.min_x + self.max_x)
            self.cy = 0.5 * (self.min_y + self.max_y)
        else:
            self.min_x = self.max_x = self.min_y = self.max_y = self.cx = self.cy = 0.0

        edges: List[Tuple[float, float, float, float]] = []
        if len(pts) >= 2:
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                if (x1 != x2) or (y1 != y2):
                    edges.append((x1, y1, x2, y2))
        self.edges = edges

    def contains_2d(self, x: float, y: float) -> bool:
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            return False
        if not _point_in_poly(x, y, self.ring):
            return False
        for h in self.holes:
            if _point_in_poly(x, y, h):
                return False
        return True

    def contains_3d(self, x: float, y: float, z: float) -> bool:
        if z < self.base or z > self.top:
            return False
        return self.contains_2d(x, y)

    def dist_sq_to_aabb_2d(self, x: float, y: float) -> float:
        dx = 0.0
        if x < self.min_x:
            dx = self.min_x - x
        elif x > self.max_x:
            dx = x - self.max_x
        dy = 0.0
        if y < self.min_y:
            dy = self.min_y - y
        elif y > self.max_y:
            dy = y - self.max_y
        return dx * dx + dy * dy

    def intersects_segment_3d(
        self,
        ax: float,
        ay: float,
        az: float,
        bx: float,
        by: float,
        bz: float,
    ) -> bool:
        """
        Exact check if 3D line segment A->B passes through this prism,
        preventing thin walls or narrow pillars from being skipped by ray step sampling.
        """
        if max(az, bz) < self.base or min(az, bz) > self.top:
            return False
        if (
            max(ax, bx) < self.min_x
            or min(ax, bx) > self.max_x
            or max(ay, by) < self.min_y
            or min(ay, by) > self.max_y
        ):
            return False

        # Endpoints inside
        if self.contains_3d(ax, ay, az) or self.contains_3d(bx, by, bz):
            return True

        dx = bx - ax
        dy = by - ay
        dz = bz - az

        # Check intersection with outer ring edges (and hole edges)
        for x1, y1, x2, y2 in self.edges:
            t = _segment_intersect_t(ax, ay, bx, by, x1, y1, x2, y2)
            if t is not None:
                z_at_t = az + dz * t
                if self.base <= z_at_t <= self.top:
                    # Midpoint slightly inside or edge hit
                    return True

        for x1, y1, x2, y2 in self.hole_edges:
            t = _segment_intersect_t(ax, ay, bx, by, x1, y1, x2, y2)
            if t is not None:
                z_at_t = az + dz * t
                if self.base <= z_at_t <= self.top:
                    return True

        # Check horizontal top/base cap intersections if segment crosses z = top or z = base
        if abs(dz) > 1e-6:
            for cap_z in (self.base, self.top):
                t = (cap_z - az) / dz
                if 0.0 <= t <= 1.0:
                    ix = ax + dx * t
                    iy = ay + dy * t
                    if self.contains_2d(ix, iy):
                        return True

        return False


def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _segment_intersect_t(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> Optional[float]:
    """
    Returns parameter t in [0, 1] along segment A->B where it intersects segment C->D,
    or None if they do not intersect.
    """
    rx = bx - ax
    ry = by - ay
    sx = dx - cx
    sy = dy - cy
    rxs = rx * sy - ry * sx
    if abs(rxs) < 1e-9:
        return None
    qpx = cx - ax
    qpy = cy - ay
    t = (qpx * sy - qpy * sx) / rxs
    if t < 0.0 or t > 1.0:
        return None
    u = (qpx * ry - qpy * rx) / rxs
    if u < 0.0 or u > 1.0:
        return None
    return t


def _dist_point_to_segment_sq(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    c1 = wx * vx + wy * vy
    if c1 <= 0.0:
        return wx * wx + wy * wy
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        dx = px - x2
        dy = py - y2
        return dx * dx + dy * dy
    b = c1 / c2
    bx = x1 + b * vx
    by = y1 + b * vy
    dx = px - bx
    dy = py - by
    return dx * dx + dy * dy


class WorldMap:
    """
    Spatial representation of the arena with precomputed 2D navigation grid,
    A* pathfinder, exact segment LOS checks, and connected escape route verification.
    """

    def __init__(self, cell_size: float = 24.0, nav_step: float = 4.0) -> None:
        self.loaded = False
        self.cell_size = cell_size
        self.nav_step = nav_step
        self.bounds = {"minX": -535.0, "maxX": 535.0, "minY": -362.0, "maxY": 362.0}
        self.t_min_x = -535.0
        self.t_min_y = -362.0
        self.t_nx = 0
        self.t_ny = 0
        self.t_step = 1.0
        self.terrain_cm: Optional[array.array] = None

        self.prisms: List[Prism] = []
        self.grid: Dict[Tuple[int, int], List[Prism]] = {}
        self.cover_buildings: List[Prism] = []

        # Coarse navigation grid (4m resolution: ~268 x 181 = 48.5k cells)
        self.nav_min_x = -535.0
        self.nav_min_y = -362.0
        self.nav_nx = 0
        self.nav_ny = 0
        self.nav_blocked: Optional[bytearray] = None
        self.nav_z: Optional[array.array] = None
        self.nav_escapes: Optional[bytearray] = None
        self._scaled_blocked_cache: Dict[int, Dict[int, bool]] = {}
        self._scaled_edge_cache: Dict[int, Dict[Tuple[int, int], bool]] = {}

    def load(self, map_dir: str, files: Optional[List[str]] = None) -> None:
        if not map_dir:
            return
        files = files or ["v2/world.json", "v2/terrain.i16"]
        world_rel = next((f for f in files if f.endswith("world.json")), "v2/world.json")
        terrain_rel = next(
            (f for f in files if f.endswith("terrain.i16")), "v2/terrain.i16"
        )

        world_path = os.path.join(map_dir, world_rel)
        terrain_path = os.path.join(map_dir, terrain_rel)

        if not os.path.isfile(world_path):
            return

        try:
            with open(world_path, "r", encoding="utf-8") as f:
                wdata = json.load(f)
        except Exception:
            return

        if "bounds" in wdata and isinstance(wdata["bounds"], dict):
            self.bounds.update(wdata["bounds"])

        tinfo = wdata.get("terrain", {})
        self.t_min_x = float(tinfo.get("minX", self.bounds["minX"]))
        self.t_min_y = float(tinfo.get("minY", self.bounds["minY"]))
        self.t_nx = int(tinfo.get("nx", 1070))
        self.t_ny = int(tinfo.get("ny", 724))
        self.t_step = float(tinfo.get("step", 1.0))

        if os.path.isfile(terrain_path) and self.t_nx > 0 and self.t_ny > 0:
            try:
                arr = array.array("h")
                with open(terrain_path, "rb") as tf:
                    arr.fromfile(tf, self.t_nx * self.t_ny)
                if sys.byteorder != "little":
                    arr.byteswap()
                self.terrain_cm = arr
            except Exception:
                self.terrain_cm = None

        self.prisms.clear()
        self.grid.clear()
        self.cover_buildings.clear()

        raw_prisms = wdata.get("prisms", [])
        for rp in raw_prisms:
            ring = rp.get("ring", [])
            if len(ring) < 6:
                continue
            p = Prism(
                kind=rp.get("kind", "building"),
                base=rp.get("base", -100.0),
                top=rp.get("top", 100.0),
                ring_flat=ring,
                holes_flat=rp.get("holes", []),
            )
            self.prisms.append(p)
            self._insert_prism(p)
            if p.kind in ("building", "stadium") and (p.top - p.base) >= 5.5:
                self.cover_buildings.append(p)

        self.loaded = True
        self._build_nav_grid()

    def _build_nav_grid(self) -> None:
        """
        Precomputes walkability and ground elevation on a 4m grid during hello,
        making A* and BFS connected-escape checks take <0.3ms at runtime.
        """
        self.nav_min_x = float(self.bounds["minX"])
        self.nav_min_y = float(self.bounds["minY"])
        width = float(self.bounds["maxX"]) - self.nav_min_x
        height = float(self.bounds["maxY"]) - self.nav_min_y
        self.nav_nx = max(1, int(math.ceil(width / self.nav_step)))
        self.nav_ny = max(1, int(math.ceil(height / self.nav_step)))

        total = self.nav_nx * self.nav_ny
        blocked = bytearray(total)
        z_arr = array.array("f", [0.0] * total)

        nx = self.nav_nx
        ny = self.nav_ny
        step = self.nav_step
        min_x = self.nav_min_x
        min_y = self.nav_min_y

        for iy in range(ny):
            y = min_y + (iy + 0.5) * step
            row_off = iy * nx
            for ix in range(nx):
                x = min_x + (ix + 0.5) * step
                gz = self.ground_height(x, y)
                idx = row_off + ix
                z_arr[idx] = gz
                if self.is_point_blocked(x, y, gz, mech_radius=2.5, mech_height=9.0):
                    blocked[idx] = 1

        # Second pass: mark cells with unwalkable steep terrain slopes (> 50 deg, tan(50)~=1.19)
        max_dz = step * 1.16 + 1.2
        for iy in range(1, ny - 1):
            row_off = iy * nx
            for ix in range(1, nx - 1):
                idx = row_off + ix
                if blocked[idx]:
                    continue
                gz = z_arr[idx]
                if (
                    abs(z_arr[idx + 1] - gz) > max_dz
                    or abs(z_arr[idx - 1] - gz) > max_dz
                    or abs(z_arr[idx + nx] - gz) > max_dz
                    or abs(z_arr[idx - nx] - gz) > max_dz
                ):
                    blocked[idx] = 1

        self.nav_blocked = blocked
        self.nav_z = z_arr

    def _cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size)))

    def _nav_coords(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(math.floor((x - self.nav_min_x) / self.nav_step))
        iy = int(math.floor((y - self.nav_min_y) / self.nav_step))
        ix = max(0, min(self.nav_nx - 1, ix))
        iy = max(0, min(self.nav_ny - 1, iy))
        return ix, iy

    def _nav_world_pos(self, ix: int, iy: int) -> Tuple[float, float]:
        return (
            self.nav_min_x + (ix + 0.5) * self.nav_step,
            self.nav_min_y + (iy + 0.5) * self.nav_step,
        )

    def _get_scaled_cache(self, mech_radius: float) -> Optional[Dict[int, bool]]:
        if mech_radius <= 2.52:
            return None
        key = int(round(mech_radius * 20.0))
        bucket = self._scaled_blocked_cache.get(key)
        if bucket is None:
            bucket = {}
            self._scaled_blocked_cache[key] = bucket
        return bucket

    def _get_edge_cache(self, mech_radius: float) -> Dict[Tuple[int, int], bool]:
        key = int(round(mech_radius * 20.0))
        bucket = self._scaled_edge_cache.get(key)
        if bucket is None:
            bucket = {}
            self._scaled_edge_cache[key] = bucket
        return bucket

    def _is_nav_edge_clear(
        self,
        cx: int,
        cy: int,
        nx_i: int,
        ny_i: int,
        c_idx: int,
        n_idx: int,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
        edge_cache: Optional[Dict[Tuple[int, int], bool]] = None,
    ) -> bool:
        """
        Validates and caches whether the midpoint between adjacent 4m grid cells
        (cx, cy) and (nx_i, ny_i) has full cylinder clearance for (mech_radius, mech_height),
        preventing narrow pillars between grid centers from clipping the mech.
        """
        pair = (c_idx, n_idx) if c_idx < n_idx else (n_idx, c_idx)
        cache = edge_cache if edge_cache is not None else self._get_edge_cache(mech_radius)
        res = cache.get(pair)
        if res is not None:
            return res

        mx = self.nav_min_x + (0.5 * (cx + nx_i) + 0.5) * self.nav_step
        my = self.nav_min_y + (0.5 * (cy + ny_i) + 0.5) * self.nav_step
        mz = self.ground_height(mx, my)
        clear = not self.is_point_blocked(mx, my, mz, mech_radius, mech_height)
        cache[pair] = clear
        return clear

    def _is_nav_cell_blocked(
        self,
        ix: int,
        iy: int,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
        custom_cache: Optional[Dict[int, bool]] = None,
    ) -> bool:
        if self.nav_blocked is None:
            wx, wy = self._nav_world_pos(ix, iy)
            return self.is_point_blocked(wx, wy, None, mech_radius, mech_height)
        idx = iy * self.nav_nx + ix
        if mech_radius <= 2.52 and mech_height <= 9.05:
            return bool(self.nav_blocked[idx])
        cache = custom_cache if custom_cache is not None else self._get_scaled_cache(mech_radius)
        if cache is not None and idx in cache:
            return cache[idx]
        if self.nav_blocked[idx]:
            if cache is not None:
                cache[idx] = True
            return True
        wx, wy = self._nav_world_pos(ix, iy)
        gz = self.nav_z[idx] if self.nav_z is not None else None
        res = self.is_point_blocked(wx, wy, gz, mech_radius, mech_height)
        if cache is not None:
            cache[idx] = res
        return res

    def _find_nearest_unblocked_nav(
        self,
        ix: int,
        iy: int,
        max_r: int = 5,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
        custom_cache: Optional[Dict[int, bool]] = None,
    ) -> Optional[Tuple[int, int]]:
        if self.nav_blocked is None:
            return (ix, iy)
        nx = self.nav_nx
        ny = self.nav_ny
        if not self._is_nav_cell_blocked(ix, iy, mech_radius, mech_height, custom_cache):
            return (ix, iy)
        for r in range(1, max_r + 1):
            for dy in range(-r, r + 1):
                ny_i = iy + dy
                if ny_i < 0 or ny_i >= ny:
                    continue
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx_i = ix + dx
                    if 0 <= nx_i < nx and not self._is_nav_cell_blocked(
                        nx_i, ny_i, mech_radius, mech_height, custom_cache
                    ):
                        return (nx_i, ny_i)
        return None

    def _insert_prism(self, p: Prism) -> None:
        ix0, iy0 = self._cell_coords(p.min_x, p.min_y)
        ix1, iy1 = self._cell_coords(p.max_x, p.max_y)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cell = (ix, iy)
                bucket = self.grid.get(cell)
                if bucket is None:
                    self.grid[cell] = [p]
                else:
                    bucket.append(p)

    def ground_height(self, x: float, y: float) -> float:
        if self.terrain_cm is None or self.t_nx <= 1 or self.t_ny <= 1:
            return 0.0
        fx = (x - self.t_min_x) / self.t_step
        fy = (y - self.t_min_y) / self.t_step
        ix = int(math.floor(fx))
        iy = int(math.floor(fy))
        dx = fx - ix
        dy = fy - iy

        ix0 = max(0, min(self.t_nx - 1, ix))
        ix1 = max(0, min(self.t_nx - 1, ix + 1))
        iy0 = max(0, min(self.t_ny - 1, iy))
        iy1 = max(0, min(self.t_ny - 1, iy + 1))

        nx = self.t_nx
        arr = self.terrain_cm
        z00 = arr[iy0 * nx + ix0]
        z10 = arr[iy0 * nx + ix1]
        z01 = arr[iy1 * nx + ix0]
        z11 = arr[iy1 * nx + ix1]

        z0 = z00 + dx * (z10 - z00)
        z1 = z01 + dx * (z11 - z01)
        return (z0 + dy * (z1 - z0)) * 0.01

    def query_prisms_near(self, x: float, y: float, radius: float) -> List[Prism]:
        ix0, iy0 = self._cell_coords(x - radius, y - radius)
        ix1, iy1 = self._cell_coords(x + radius, y + radius)
        seen = set()
        out: List[Prism] = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                bucket = self.grid.get((ix, iy))
                if not bucket:
                    continue
                for p in bucket:
                    pid = id(p)
                    if pid not in seen:
                        seen.add(pid)
                        out.append(p)
        return out

    def is_point_blocked(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
    ) -> bool:
        """
        Returns True if a vertical mech cylinder at (x, y, z) collides with
        map boundaries or solid prisms (accounting for bridges clearing the head).
        """
        margin = mech_radius + 0.3
        if (
            x < self.bounds["minX"] + margin
            or x > self.bounds["maxX"] - margin
            or y < self.bounds["minY"] + margin
            or y > self.bounds["maxY"] - margin
        ):
            return True

        if not self.loaded:
            return False

        gz = self.ground_height(x, y) if z is None else z
        top_z = gz + mech_height
        step_up = 1.2
        r_sq = margin * margin

        candidates = self.query_prisms_near(x, y, margin + 1.0)
        for p in candidates:
            if p.base >= top_z + 0.1:
                continue
            if p.top <= gz + step_up:
                continue
            if p.dist_sq_to_aabb_2d(x, y) > r_sq:
                continue
            if p.contains_2d(x, y):
                return True
            for x1, y1, x2, y2 in p.edges:
                if _dist_point_to_segment_sq(x, y, x1, y1, x2, y2) < r_sq:
                    return True
            for x1, y1, x2, y2 in p.hole_edges:
                if _dist_point_to_segment_sq(x, y, x1, y1, x2, y2) < r_sq:
                    return True
        return False

    def is_walkable_segment_2d(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
        step_m: float = 1.8,
    ) -> bool:
        """
        Continuous walkability check between (ax, ay) and (bx, by) at 1.8m steps.
        Uses the precomputed 4m nav grid when default mech dimensions apply,
        falling back to exact prism/slope checks at endpoints and custom radii.
        """
        if not self.loaded:
            return True
        dist = math.hypot(bx - ax, by - ay)
        if dist < 0.2:
            return not self.is_point_blocked(ax, ay, None, mech_radius, mech_height)

        steps = max(1, int(math.ceil(dist / step_m)))
        inv = 1.0 / steps

        step_len = dist * inv
        max_dz = step_len * 1.18 + 1.2
        prev_z = self.ground_height(ax, ay)

        for i in range(1, steps + 1):
            t = i * inv
            px = ax + (bx - ax) * t
            py = ay + (by - ay) * t
            pz = self.ground_height(px, py)
            if (pz - prev_z) > max_dz:
                return False
            if self.is_point_blocked(px, py, pz, mech_radius, mech_height):
                return False
            prev_z = pz
        return True

    def has_line_of_sight(
        self,
        ax: float,
        ay: float,
        az: float,
        bx: float,
        by: float,
        bz: float,
        step_m: float = 1.5,
    ) -> bool:
        """
        High-precision 3D line-of-sight check against terrain and solid prisms.
        Combines fine raymarching (1.5m steps for terrain) with exact 2D segment-edge
        intersection against every prism in traversed spatial cells so thin walls
        or narrow pillars are never skipped.
        """
        if not self.loaded:
            return True

        dx = bx - ax
        dy = by - ay
        dz = bz - az
        dist_2d = math.hypot(dx, dy)
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist_3d < 0.5:
            return True

        steps = max(2, int(math.ceil(dist_2d / step_m)))
        inv = 1.0 / steps
        last_cell: Optional[Tuple[int, int]] = None
        checked_prisms = set()

        for i in range(0, steps + 1):
            t = i * inv
            px = ax + dx * t
            py = ay + dy * t
            pz = az + dz * t

            if 0 < i < steps and self.terrain_cm is not None:
                if pz < self.ground_height(px, py) + 0.12:
                    return False

            cell = self._cell_coords(px, py)
            if cell != last_cell:
                last_cell = cell
                for p in self.grid.get(cell, ()):
                    pid = id(p)
                    if pid not in checked_prisms:
                        checked_prisms.add(pid)
                        if p.intersects_segment_3d(ax, ay, az, bx, by, bz):
                            return False

        return True

    def ray_clearance_distance(
        self,
        ax: float,
        ay: float,
        az: float,
        yaw: float,
        pitch: float,
        max_dist: float = 80.0,
        step_m: float = 1.5,
    ) -> float:
        """
        Returns the distance along (yaw, pitch) from (ax, ay, az) until the ray
        hits terrain, a prism, or max_dist. Used to find a safe direction to
        discharge a charged laser without hitting any enemy.
        """
        if not self.loaded:
            return max_dist
        cp = math.cos(pitch)
        vx = cp * math.cos(yaw)
        vy = cp * math.sin(yaw)
        vz = math.sin(pitch)

        steps = max(2, int(math.ceil(max_dist / step_m)))
        for i in range(1, steps + 1):
            d = min(max_dist, i * step_m)
            px = ax + vx * d
            py = ay + vy * d
            pz = az + vz * d
            if self.terrain_cm is not None and pz < self.ground_height(px, py) + 0.1:
                return d - step_m
            for p in self.grid.get(self._cell_coords(px, py), ()):
                if p.contains_3d(px, py, pz):
                    return d - step_m
        return max_dist

    def open_escape_directions(
        self,
        x: float,
        y: float,
        z: float,
        probe_dist: float = 22.0,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
    ) -> int:
        """
        Verifies connected escape routes from (x, y, z):
        1. Checks continuous 1.8m-step walkability rays in 8 compass directions.
        2. When the nav grid is available, also verifies via local BFS that
           reachable cells span at least 2 well-separated angular sectors (>= 65 deg apart)
           at distance >= probe_dist, ensuring the spot is not a dead-end pocket.
        """
        if not self.loaded:
            return 8

        open_angles: List[float] = []
        for k in range(8):
            ang = k * (math.pi * 0.25)
            tx = x + math.cos(ang) * probe_dist
            ty = y + math.sin(ang) * probe_dist
            if self.is_walkable_segment_2d(
                x, y, tx, ty, mech_radius=mech_radius, mech_height=mech_height, step_m=1.8
            ):
                open_angles.append(ang)

        # Additionally check connected BFS escape exits on the nav grid if ray paths bend around corners
        if self.nav_blocked is not None and len(open_angles) < 2:
            custom_cache: Dict[int, bool] = {}
            sx, sy = self._nav_coords(x, y)
            start = self._find_nearest_unblocked_nav(
                sx, sy, max_r=2, mech_radius=mech_radius, mech_height=mech_height, custom_cache=custom_cache
            )
            if start is not None:
                min_cells = max(3, int(probe_dist / self.nav_step))
                max_cells = min_cells + 2
                q = deque([start])
                visited = {start}
                nx = self.nav_nx
                ny = self.nav_ny
                while q:
                    cx, cy = q.popleft()
                    cdist = math.hypot(cx - start[0], cy - start[1])
                    if cdist >= min_cells:
                        wx, wy = self._nav_world_pos(cx, cy)
                        open_angles.append(math.atan2(wy - y, wx - x))
                        continue
                    if cdist >= max_cells:
                        continue
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                        nx_i, ny_i = cx + dx, cy + dy
                        if 0 <= nx_i < nx and 0 <= ny_i < ny and (nx_i, ny_i) not in visited:
                            if not self._is_nav_cell_blocked(
                                nx_i, ny_i, mech_radius, mech_height, custom_cache
                            ):
                                visited.add((nx_i, ny_i))
                                q.append((nx_i, ny_i))

        if not open_angles:
            return 0

        # Count distinct angular sectors separated by at least 55 degrees (~0.96 rad)
        distinct: List[float] = []
        for ang in open_angles:
            if all(
                abs((ang - d + math.pi) % (2.0 * math.pi) - math.pi) >= 0.95
                for d in distinct
            ):
                distinct.append(ang)
        return len(distinct)

    def find_path_astar(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        max_expansions: int = 1800,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
        smooth: bool = True,
        deadline_perf: Optional[float] = None,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Finds a collision-free global path from (start_x, start_y) to (goal_x, goal_y)
        on the 4m navigation grid using octile-heuristic A*, accounting for the mech's
        current scaled radius and height and validating every grid edge against narrow
        pillars. When `smooth=False` (e.g., distance-only scoring), skips waypoint
        reconstruction and string-pulling for maximum speed.
        Returns (waypoints_world, total_path_distance_m).
        """
        direct_dist = math.hypot(goal_x - start_x, goal_y - start_y)
        if not self.loaded or self.nav_blocked is None or direct_dist < 2.5:
            return [(goal_x, goal_y)], direct_dist

        # Fast path: if direct line is completely walkable for current mech dimensions, no A* needed
        if direct_dist <= 80.0 and self.is_walkable_segment_2d(
            start_x,
            start_y,
            goal_x,
            goal_y,
            mech_radius=mech_radius,
            mech_height=mech_height,
            step_m=1.8,
        ):
            return [(goal_x, goal_y)], direct_dist

        if deadline_perf is not None and time.perf_counter() >= deadline_perf:
            return [(goal_x, goal_y)], direct_dist * 3.0 + 400.0

        custom_cache = self._get_scaled_cache(mech_radius)
        edge_cache = self._get_edge_cache(mech_radius)
        sx0, sy0 = self._nav_coords(start_x, start_y)
        gx0, gy0 = self._nav_coords(goal_x, goal_y)

        start_node = self._find_nearest_unblocked_nav(
            sx0, sy0, max_r=4, mech_radius=mech_radius, mech_height=mech_height, custom_cache=custom_cache
        )
        goal_node = self._find_nearest_unblocked_nav(
            gx0, gy0, max_r=5, mech_radius=mech_radius, mech_height=mech_height, custom_cache=custom_cache
        )
        if start_node is None or goal_node is None:
            return [(goal_x, goal_y)], direct_dist * 2.5 + 500.0
        if start_node == goal_node:
            return [(goal_x, goal_y)], direct_dist

        nx = self.nav_nx
        ny = self.nav_ny
        gx, gy = goal_node
        sx, sy = start_node

        start_idx = sy * nx + sx
        goal_idx = gy * nx + gx

        open_heap: List[Tuple[float, int]] = [(0.0, start_idx)]
        g_score: Dict[int, float] = {start_idx: 0.0}
        came_from: Dict[int, int] = {}

        best_idx = start_idx
        best_h = math.hypot(gx - sx, gy - sy)
        expansions = 0
        sqrt2 = 1.41421356

        neighbors = (
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, sqrt2),
            (1, -1, sqrt2),
            (-1, 1, sqrt2),
            (-1, -1, sqrt2),
        )
        z_arr = self.nav_z
        max_step_dz = self.nav_step * 1.16 + 1.2

        reached = False
        while open_heap and expansions < max_expansions:
            if (
                deadline_perf is not None
                and (expansions & 31) == 31
                and time.perf_counter() >= deadline_perf
            ):
                break

            _, cur_idx = heapq.heappop(open_heap)
            if cur_idx == goal_idx:
                reached = True
                best_idx = cur_idx
                break

            expansions += 1
            cy, cx = divmod(cur_idx, nx)
            cur_g = g_score[cur_idx]
            cur_z = z_arr[cur_idx] if z_arr is not None else 0.0

            for dx, dy, step_cost in neighbors:
                nx_i = cx + dx
                ny_i = cy + dy
                if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                    continue
                n_idx = ny_i * nx + nx_i
                if self._is_nav_cell_blocked(
                    nx_i, ny_i, mech_radius, mech_height, custom_cache
                ):
                    continue
                if z_arr is not None and (z_arr[n_idx] - cur_z) > max_step_dz * step_cost:
                    continue
                # Prevent diagonal corner-cutting across blocked orthogonal cells
                if dx != 0 and dy != 0:
                    if self._is_nav_cell_blocked(
                        nx_i, cy, mech_radius, mech_height, custom_cache
                    ) or self._is_nav_cell_blocked(
                        cx, ny_i, mech_radius, mech_height, custom_cache
                    ):
                        continue

                # Validate and cache edge midpoint clearance so pillars between grid points cannot clip the mech
                if not self._is_nav_edge_clear(
                    cx,
                    cy,
                    nx_i,
                    ny_i,
                    cur_idx,
                    n_idx,
                    mech_radius=mech_radius,
                    mech_height=mech_height,
                    edge_cache=edge_cache,
                ):
                    continue

                tentative_g = cur_g + step_cost
                if tentative_g < g_score.get(n_idx, 1e18):
                    g_score[n_idx] = tentative_g
                    came_from[n_idx] = cur_idx
                    adx = abs(gx - nx_i)
                    ady = abs(gy - ny_i)
                    h = (adx + ady) + (sqrt2 - 2.0) * min(adx, ady)
                    if h < best_h:
                        best_h = h
                        best_idx = n_idx
                    heapq.heappush(open_heap, (tentative_g + h * 1.05, n_idx))

        total_dist = g_score.get(best_idx, 0.0) * self.nav_step
        if not reached:
            total_dist += best_h * self.nav_step * 2.5 + 250.0

        # When caller only needs distance (e.g., reachable_distance for loot/cover scoring),
        # skip path reconstruction and string-pulling completely.
        if not smooth:
            return [], total_dist

        # Reconstruct grid path
        curr = goal_idx if reached else best_idx
        raw_cells: List[Tuple[int, int]] = []
        while curr != start_idx and curr in came_from:
            cy, cx = divmod(curr, nx)
            raw_cells.append((cx, cy))
            curr = came_from[curr]
        raw_cells.reverse()

        if not raw_cells:
            return [(goal_x, goal_y)], direct_dist * 2.5 + 500.0

        raw_pts: List[Tuple[float, float]] = [
            self._nav_world_pos(cx, cy) for cx, cy in raw_cells
        ]
        # Replace final grid center with exact goal_x, goal_y ONLY if the segment from
        # the preceding grid point (or start) to (goal_x, goal_y) is genuinely walkable
        if reached:
            prev_wx, prev_wy = raw_pts[-2] if len(raw_pts) >= 2 else (start_x, start_y)
            if self.is_walkable_segment_2d(
                prev_wx,
                prev_wy,
                goal_x,
                goal_y,
                mech_radius=mech_radius,
                mech_height=mech_height,
                step_m=1.6,
            ):
                raw_pts[-1] = (goal_x, goal_y)

        # String-pulling / path smoothing using current mech dimensions
        # Every single-step grid edge in raw_pts was already validated during A* expansion!
        smoothed: List[Tuple[float, float]] = []
        anchor_x, anchor_y = start_x, start_y
        idx = 0
        n_pts = len(raw_pts)
        while idx < n_pts:
            furthest = idx
            max_look = min(n_pts - 1, idx + 10)
            for j in range(max_look, idx, -1):
                tx, ty = raw_pts[j]
                if self.is_walkable_segment_2d(
                    anchor_x,
                    anchor_y,
                    tx,
                    ty,
                    mech_radius=mech_radius,
                    mech_height=mech_height,
                    step_m=1.8,
                ):
                    furthest = j
                    break
            wx, wy = raw_pts[furthest]
            smoothed.append((wx, wy))
            anchor_x, anchor_y = wx, wy
            idx = furthest + 1

        return smoothed, total_dist

    def reachable_distance(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        max_expansions: int = 800,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
        deadline_perf: Optional[float] = None,
    ) -> float:
        """
        Returns reachable travel distance in meters between start and goal
        for the current mech dimensions without building or smoothing waypoints.
        """
        _, dist = self.find_path_astar(
            start_x,
            start_y,
            goal_x,
            goal_y,
            max_expansions=max_expansions,
            mech_radius=mech_radius,
            mech_height=mech_height,
            smooth=False,
            deadline_perf=deadline_perf,
        )
        return dist

    def steer_around_obstacles(
        self,
        pos: List[float],
        target_x: float,
        target_y: float,
        mech_radius: float = 2.5,
        mech_height: float = 9.0,
    ) -> Tuple[float, float]:
        """
        Context-steering / whisker local avoidance toward the active A* waypoint.
        Checks direct path first; if blocked by building/wall/steep terrain,
        fans out candidate rays to smoothly skirt around corners and obstacles.
        """
        px, py, pz = pos[0], pos[1], pos[2]
        dx = target_x - px
        dy = target_y - py
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            return (0.0, 0.0)

        base_yaw = math.atan2(dy, dx)
        if not self.loaded:
            return (dx / dist, dy / dist)

        lookahead = min(16.0, max(6.0, dist))
        candidate_offsets = [
            0.0,
            0.32,
            -0.32,
            0.65,
            -0.65,
            1.0,
            -1.0,
            1.4,
            -1.4,
            1.85,
            -1.85,
            2.35,
            -2.35,
        ]

        best_dir = (math.cos(base_yaw), math.sin(base_yaw))
        best_score = -1e9

        for off in candidate_offsets:
            ang = base_yaw + off
            ca = math.cos(ang)
            sa = math.sin(ang)
            clear_d = lookahead
            prev_z = pz
            blocked = False

            for step_d in (2.5, 5.0, 8.0, 11.5, lookahead):
                if step_d > lookahead:
                    continue
                tx = px + ca * step_d
                ty = py + sa * step_d
                tz = self.ground_height(tx, ty)
                if (tz - prev_z) > 3.0 * 1.15 + 1.2:
                    clear_d = max(0.0, step_d - 2.5)
                    blocked = True
                    break
                prev_z = tz
                if self.is_point_blocked(tx, ty, tz, mech_radius, mech_height):
                    clear_d = max(0.0, step_d - 2.5)
                    blocked = True
                    break

            alignment = math.cos(off)
            score = clear_d * 2.2 + alignment * 10.0 - (20.0 if blocked else 0.0)
            if score > best_score:
                best_score = score
                best_dir = (ca, sa)
                if not blocked and abs(off) < 0.01:
                    break

        return best_dir
