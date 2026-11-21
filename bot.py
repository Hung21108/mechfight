#!/usr/bin/env python3
"""
Mobile Scavenger Bot for mechfight/1.

Core loop:
  collect safe power -> secure future cover -> take favorable shots ->
  break contact and heal -> repeat (switching to lightning when sustained
  close-range encounters and reachable pickups favor it).
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from mechfight import DebugDraw, log, run_bot
from world import WorldMap


def wrap_angle(a: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    """Signed shortest difference a - b in [-pi, pi]."""
    return wrap_angle(a - b)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def yaw_pitch_to(
    src_x: float, src_y: float, src_z: float, dst_x: float, dst_y: float, dst_z: float
) -> Tuple[float, float]:
    dx = dst_x - src_x
    dy = dst_y - src_y
    dz = dst_z - src_z
    horiz = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, max(1e-6, horiz))
    return yaw, pitch


def solve_intercept_3d(
    shooter_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    target_vel: Tuple[float, float, float],
    proj_speed: float,
    shot_origin_offset: float = 3.0,
) -> Tuple[float, float, float, float]:
    """
    Solves quadratic intercept for a straight-line projectile of speed `proj_speed`
    leaving `shot_origin_offset` (3.0m) along the aim vector.
    Returns (aim_x, aim_y, aim_z, flight_time).
    """
    rx = target_pos[0] - shooter_pos[0]
    ry = target_pos[1] - shooter_pos[1]
    rz = target_pos[2] - shooter_pos[2]
    vx, vy, vz = target_vel

    dist = math.sqrt(rx * rx + ry * ry + rz * rz)
    if proj_speed <= 1.0 or dist < 1.0:
        return target_pos[0], target_pos[1], target_pos[2], 0.0

    t = max(0.0, (dist - shot_origin_offset) / proj_speed)
    for _ in range(3):
        px = rx + vx * t
        py = ry + vy * t
        pz = rz + vz * t
        d = math.sqrt(px * px + py * py + pz * pz)
        t = max(0.0, (d - shot_origin_offset) / proj_speed)

    return (
        target_pos[0] + vx * t,
        target_pos[1] + vy * t,
        target_pos[2] + vz * t,
        t,
    )


class EnemyTrack:
    """Short-term memory of an opponent or sound threat."""

    __slots__ = (
        "id",
        "pos",
        "vel",
        "aim",
        "hp",
        "max_hp",
        "cubes",
        "weapons",
        "charging",
        "last_seen_t",
    )

    def __init__(self, m: Dict[str, Any], t: float) -> None:
        self.id: int = int(m["id"])
        self.pos: List[float] = list(m["pos"])
        self.vel: List[float] = list(m.get("vel", [0.0, 0.0, 0.0]))
        self.aim: List[float] = list(m.get("aim", [0.0, 0.0]))
        self.hp: float = float(m.get("hp", 1400.0))
        self.max_hp: float = float(m.get("max_hp", 1400.0))
        self.cubes: int = int(m.get("cubes", 0))
        self.weapons: List[str] = list(m.get("weapons", ["gun"]))
        self.charging: bool = bool(m.get("charging", False))
        self.last_seen_t: float = t


class SoundPing:
    __slots__ = ("kind", "bearing", "dist_band", "est_dist", "t")

    def __init__(self, kind: str, bearing: float, dist_band: str, t: float) -> None:
        self.kind = kind
        self.bearing = bearing
        self.dist_band = dist_band
        if dist_band == "near":
            self.est_dist = 35.0
        elif dist_band == "mid":
            self.est_dist = 105.0
        else:
            self.est_dist = 220.0
        self.t = t


class MobileScavengerBot:
    def __init__(
        self,
        enable_finishable_targeting: bool = True,
        enable_early_multi_reposition: bool = True,
    ) -> None:
        self.enable_finishable_targeting = enable_finishable_targeting
        self.enable_early_multi_reposition = enable_early_multi_reposition
        self.slot: int = 0
        self.seed: int = 1
        self.rng = random.Random(1)
        self.config: Dict[str, Any] = {}
        self.world = WorldMap()

        # Healing & combat timers (tracked via both damage taken and hits dealt)
        self.last_damaged_t: float = -100.0
        self.last_dealt_damage_t: float = -100.0
        self.last_combat_t: float = -100.0
        self.last_damaged_bearing: Optional[float] = None
        self.last_damaged_by: Optional[int] = None
        self.shorted_until_t: float = -100.0

        self.enemy_tracks: Dict[int, EnemyTrack] = {}
        self.sound_pings: List[SoundPing] = []

        # High-level mode: "SCAVENGE", "COMBAT", "RETREAT", "DROP"
        self.mode: str = "SCAVENGE"
        # Retreat sub-state: "HEALING", "ESCAPING", "CORNERED"
        self.retreat_substate: str = "ESCAPING"
        # Laser state machine: "IDLE", "CHARGING", "RELEASING_AT_TARGET", "HOLDING_FOR_RELEASE", "DISCHARGING_SAFE"
        self.laser_state: str = "IDLE"

        # Cover anchor & A* path planner state
        self.anchor_pos: Optional[Tuple[float, float]] = None
        self.anchor_refresh_t: float = -100.0
        self.current_path: List[Tuple[float, float]] = []
        self.path_goal: Optional[Tuple[float, float]] = None
        self.path_refresh_t: float = -100.0

        # Stuck detection & recovery
        self.progress_history: List[Tuple[float, float, float, float]] = []
        self.stuck_until_t: float = -100.0
        self.unstuck_dir: Tuple[float, float] = (0.0, 0.0)
        self.unstuck_side: int = 1
        self.blacklisted_goals: List[Tuple[float, float, float]] = []

        # Shared planning deadline across all reachability & pathfinding queries in a single tick
        self.tick_deadline_perf: Optional[float] = None

        # Encounter history & weapon-preference hysteresis
        self.recent_combat_dists: List[float] = []
        self.preferred_secondary: str = "laser"
        self.last_pref_switch_t: float = -100.0

    # ------------------------------------------------------------------
    # Lifecycle handlers
    # ------------------------------------------------------------------

    def on_hello(self, msg: Dict[str, Any]) -> None:
        self.slot = int(msg.get("slot", 0))
        self.seed = int(msg.get("seed", 1))
        self.rng = random.Random(self.seed)
        self.config = msg.get("config", {})

        map_info = msg.get("map", {})
        map_dir = map_info.get("dir", "/arena")
        map_files = map_info.get("files", ["v2/world.json", "v2/terrain.i16"])
        self.world.load(map_dir, map_files)
        log(
            f"[hello] slot={self.slot} map_loaded={self.world.loaded} "
            f"prisms={len(self.world.prisms)} cover={len(self.world.cover_buildings)}"
        )

    def on_end(self, msg: Dict[str, Any]) -> None:
        log(f"[end] place={msg.get('place')} kills={msg.get('kills')}")

    # ------------------------------------------------------------------
    # Config & Self Helpers
    # ------------------------------------------------------------------

    def get_regen_delay(self, self_data: Dict[str, Any]) -> float:
        stats = self_data.get("stats", {})
        if "regen_delay" in stats:
            return float(stats["regen_delay"])
        mech_cfg = self.config.get("mech", {})
        base = float(mech_cfg.get("regen_delay", 5.0))
        upgrades = self_data.get("upgrades", [])
        nanite_stacks = sum(
            1 for u in upgrades if "nanite" in str(u).lower() or "regen" in str(u).lower()
        )
        for _ in range(nanite_stacks):
            base *= 0.63
        return max(1.5, base)

    def get_mech_dimensions(self, self_data: Dict[str, Any]) -> Tuple[float, float, float, float]:
        cubes = int(self_data.get("cubes", 0))
        cubes_cfg = self.config.get("cubes", {})
        size_per_cube = float(cubes_cfg.get("size_per_cube", 0.05))
        scale = 1.0 + size_per_cube * cubes

        stats = self_data.get("stats", {})
        radius = float(stats.get("radius", 2.5 * scale))
        height = float(stats.get("height", 9.0 * scale))
        eye_h = float(stats.get("eye_height", 8.0 * scale))
        muzzle_h = float(stats.get("muzzle_height", 7.0 * scale))
        return radius, height, eye_h, muzzle_h

    def get_weapon_slot(
        self, self_data: Dict[str, Any], kind: str
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        weapons = self_data.get("weapons", [None, None])
        for idx, w in enumerate(weapons):
            if w and w.get("kind") == kind:
                return idx, w
        return None, None

    # ------------------------------------------------------------------
    # Threat Selection (#1: Store and compare complete best threat score)
    # ------------------------------------------------------------------

    def get_enemy_weapon_reach(self, e: EnemyTrack) -> float:
        """
        Returns the maximum effective weapon range of enemy `e` based on the
        weapons they hold (reading live weapon ranges from `hello.config`).
        """
        w_cfg = self.config.get("weapons", {})
        max_reach = 0.0
        weapons = e.weapons if e.weapons else ["gun"]
        for wkind in weapons:
            if wkind == "laser":
                r = float(w_cfg.get("laser", {}).get("range", 280.0))
            elif wkind == "mortar":
                mspeed = float(w_cfg.get("mortar", {}).get("speed", 72.0))
                mgrav = max(1.0, float(w_cfg.get("mortar", {}).get("gravity", 22.0)))
                r = (mspeed * mspeed) / mgrav
            elif wkind == "gun":
                r = float(w_cfg.get("gun", {}).get("range", 230.0))
            elif wkind == "lightning":
                r = float(w_cfg.get("lightning", {}).get("range", 45.0))
            else:
                r = 230.0
            if r > max_reach:
                max_reach = r
        return max(45.0, max_reach)

    def select_primary_threat(
        self,
        self_data: Dict[str, Any],
        active_enemies: List[EnemyTrack],
        t: float,
    ) -> Tuple[Optional[EnemyTrack], float, List[EnemyTrack]]:
        """
        Evaluates every tracked enemy and selects the highest-scoring threat
        independently of list order. Also collects the exact list of `shooters`
        (enemies with clear line of sight and within their actual weapon reach)
        to reuse across repositioning, boosting, and healing decisions.
        Returns (primary_enemy, dist_to_primary, shooters_with_los).
        """
        px, py, pz = self_data["pos"]
        _, _, eye_h, _ = self.get_mech_dimensions(self_data)

        primary_enemy: Optional[EnemyTrack] = None
        primary_enemy_dist: float = 1e9
        best_threat_score: float = -1e9
        shooters: List[EnemyTrack] = []

        for e in active_enemies:
            ed = math.hypot(e.pos[0] - px, e.pos[1] - py)
            has_los = self.world.has_line_of_sight(
                px, py, pz + eye_h, e.pos[0], e.pos[1], e.pos[2] + 5.0
            )
            reach = self.get_enemy_weapon_reach(e)
            can_shoot_us = has_los and (ed <= reach)
            if can_shoot_us:
                shooters.append(e)

            # Check if enemy torso aim is pointed roughly toward us
            aim_to_us = math.atan2(py - e.pos[1], px - e.pos[0])
            aiming_at_us = abs(angle_diff(e.aim[0], aim_to_us)) < 0.45

            threat_score = 300.0 - ed
            if can_shoot_us:
                threat_score += 75.0
            elif has_los:
                threat_score += 35.0
            if e.charging:
                threat_score += 55.0 if has_los else 30.0
            if aiming_at_us and can_shoot_us:
                threat_score += 25.0
            if self.last_damaged_by == e.id and (t - self.last_damaged_t) < 4.5:
                threat_score += 70.0
            if "lightning" in e.weapons and ed < 55.0:
                threat_score += 40.0

            # Deterministic tie-break on enemy id so order in active_enemies never matters
            if (threat_score > best_threat_score + 1e-6) or (
                abs(threat_score - best_threat_score) <= 1e-6
                and primary_enemy is not None
                and e.id < primary_enemy.id
            ):
                best_threat_score = threat_score
                primary_enemy = e
                primary_enemy_dist = ed

        return primary_enemy, primary_enemy_dist, shooters

    def estimate_time_to_kill(
        self,
        self_data: Dict[str, Any],
        target: EnemyTrack,
        t: float,
    ) -> float:
        """
        Simulates a discrete weapon damage timeline against `target` to compute
        the exact time (in seconds from now) when lethal damage can actually LAND
        on the target, accounting for:
          - Torso yaw & pitch slew delay (`turn_rate`, `pitch_rate`)
          - Gun current cooldown, magazine ammo, full 2.4s reload delay when mag empties,
            and bullet flight time (`(dist - 3.0) / speed`)
          - Laser current cooldown, remaining charge time to reach required charge,
            and cook-off multiplier (+60% against mortar carriers)
          - Lightning cooldown, range, and energy availability
          - Rear-armor multiplier (1.5x if we are outside the target's 180-deg front arc).
        """
        px, py, pz = self_data["pos"]
        radius, _, _, muzzle_h = self.get_mech_dimensions(self_data)
        mz = pz + muzzle_h
        actual_yaw, actual_pitch = self_data.get("aim", [0.0, 0.0])

        tx, ty, tz = target.pos[0], target.pos[1], target.pos[2] + 4.5
        dist = math.hypot(tx - px, ty - py)

        my_cubes = int(self_data.get("cubes", 0))
        cube_cfg = self.config.get("cubes", {})
        dmg_per_cube = float(cube_cfg.get("damage_per_cube", 0.04))
        stat_cap = int(cube_cfg.get("stat_cap", 10))
        cube_mult = 1.0 + dmg_per_cube * min(my_cubes, stat_cap)

        angle_from_enemy_to_us = math.atan2(py - ty, px - tx)
        hitting_enemy_rear = abs(angle_diff(target.aim[0], angle_from_enemy_to_us)) > (
            math.pi * 0.5
        )
        rear_mult = 1.5 if hitting_enemy_rear else 1.0
        total_mult = cube_mult * rear_mult

        # Torso slew delay before our first accurate shot can leave the barrel
        stats = self_data.get("stats", {})
        mech_cfg = self.config.get("mech", {})
        turn_rate = max(0.5, float(stats.get("turn_rate", mech_cfg.get("turn_rate", 4.0))))
        pitch_rate = max(0.5, float(stats.get("pitch_rate", mech_cfg.get("pitch_rate", 3.0))))

        target_yaw, target_pitch = yaw_pitch_to(px, py, mz, tx, ty, tz)
        yaw_tol = max(0.025, math.atan2(2.2, max(5.0, dist)))
        yaw_slew_s = max(0.0, abs(angle_diff(actual_yaw, target_yaw)) - yaw_tol) / turn_rate
        pitch_slew_s = max(0.0, abs(actual_pitch - target_pitch) - 0.04) / pitch_rate
        slew_delay_s = max(yaw_slew_s, pitch_slew_s)

        # Build list of (impact_time_s, damage_amount) events up to 15.0s
        events: List[Tuple[float, float]] = []
        horizon = 15.0

        # 1. Gun shot schedule (including reloads and projectile flight time)
        _, gun_w = self.get_weapon_slot(self_data, "gun")
        if gun_w is not None:
            g_stats = gun_w.get("stats", {})
            g_range = float(g_stats.get("range", 230.0))
            if dist <= g_range:
                g_dmg = float(g_stats.get("damage", 45.0)) * total_mult
                g_cd = max(0.08, float(g_stats.get("cooldown", 0.24)))
                g_speed = max(20.0, float(g_stats.get("speed", 252.0)))
                g_mag = max(1, int(g_stats.get("mag", 12)))
                g_reload_dur = max(0.2, float(g_stats.get("reload", 2.4)))

                cur_cd = float(gun_w.get("cooldown", 0.0))
                cur_reload = float(gun_w.get("reload", 0.0))
                cur_ammo = int(gun_w.get("ammo", g_mag))
                flight_s = max(0.0, (dist - (radius + 0.5)) / g_speed)

                if cur_reload > 0.0:
                    fire_t = max(slew_delay_s, cur_reload)
                    cur_ammo = g_mag
                elif cur_ammo <= 0:
                    fire_t = max(slew_delay_s, g_reload_dur)
                    cur_ammo = g_mag
                else:
                    fire_t = max(slew_delay_s, cur_cd)

                while fire_t <= horizon:
                    impact_t = fire_t + flight_s
                    events.append((impact_t, g_dmg))
                    cur_ammo -= 1
                    if cur_ammo <= 0:
                        fire_t += g_reload_dur
                        cur_ammo = g_mag
                    else:
                        fire_t += g_cd

        # 2. Laser beam schedule (hitscan: flight_s = 0, includes charge time & cook-off)
        _, laser_w = self.get_weapon_slot(self_data, "laser")
        if laser_w is not None:
            l_stats = laser_w.get("stats", {})
            l_range = float(l_stats.get("range", 280.0))
            if dist <= l_range:
                l_dmg = float(l_stats.get("damage", 315.0)) * total_mult
                if "mortar" in target.weapons:
                    l_dmg *= 1.6
                l_full_charge = max(0.15, float(l_stats.get("charge", 0.75)))
                l_cd_dur = max(0.2, float(l_stats.get("cooldown", 1.65)))
                cur_l_cd = float(laser_w.get("cooldown", 0.0))
                cur_l_charge = float(laser_w.get("charge", 0.0))
                short_rem = max(0.0, self.shorted_until_t - t)

                # First beam release time
                req_charge = l_full_charge * 0.90
                if cur_l_charge > 0.0 and short_rem <= 0.0:
                    rem_charge = max(0.0, req_charge - cur_l_charge)
                    # Note: releasing requires 1 decision tick (0.1s) after charge completes
                    fire_t = max(slew_delay_s, rem_charge + 0.05)
                else:
                    start_charge_t = max(cur_l_cd, short_rem)
                    fire_t = max(slew_delay_s, start_charge_t + req_charge + 0.05)

                while fire_t <= horizon:
                    events.append((fire_t, l_dmg))
                    fire_t += l_cd_dur + req_charge + 0.05

        # 3. Lightning arc schedule (instant hitscan within 45m, gated by energy)
        _, light_w = self.get_weapon_slot(self_data, "lightning")
        if light_w is not None:
            lt_stats = light_w.get("stats", {})
            lt_range = float(lt_stats.get("range", 45.0))
            if dist <= lt_range:
                lt_dmg = float(lt_stats.get("damage", 60.0)) * total_mult
                lt_cd_dur = max(0.1, float(lt_stats.get("cooldown", 0.25)))
                lt_cost = max(1.0, float(lt_stats.get("energy", 3.0)))
                cur_lt_cd = float(light_w.get("cooldown", 0.0))
                avail_energy = max(0.0, float(self_data.get("energy", 100.0)) - 12.0)
                max_shots = int(avail_energy // lt_cost)

                lt_cone_rad = math.radians(float(lt_stats.get("cone_deg", 30.0)))
                lt_slew_s = (
                    max(0.0, abs(angle_diff(actual_yaw, target_yaw)) - lt_cone_rad * 0.8)
                    / turn_rate
                )
                fire_t = max(lt_slew_s, cur_lt_cd)
                shots_fired = 0
                while fire_t <= horizon and shots_fired < max_shots:
                    events.append((fire_t, lt_dmg))
                    shots_fired += 1
                    fire_t += lt_cd_dur

        if not events:
            return 999.0

        events.sort(key=lambda item: item[0])
        acc_dmg = 0.0
        target_hp = max(1.0, float(target.hp))
        for impact_t, dmg in events:
            acc_dmg += dmg
            if acc_dmg >= target_hp:
                return impact_t

        # If target survives the 15s horizon, extrapolate from total damage over horizon
        if acc_dmg <= 1.0:
            return 999.0
        return horizon * (target_hp / acc_dmg)

    def select_firing_target(
        self,
        self_data: Dict[str, Any],
        active_enemies: List[EnemyTrack],
        primary_threat: Optional[EnemyTrack],
        t: float,
    ) -> Tuple[Optional[EnemyTrack], float, bool]:
        """
        Separates the enemy we shoot (`firing_target`) from the enemy we defend against (`primary_threat`).
        Prefers the exposed opponent we can finish fastest based on actual simulated time-to-kill
        (including ammo, reloads, cooldowns, torso slew, and projectile flight time).
        Returns (firing_target, estimated_time_to_kill_s, is_kill_imminent).
        """
        if not active_enemies:
            return None, 999.0, False

        px, py, pz = self_data["pos"]
        _, _, _, muzzle_h = self.get_mech_dimensions(self_data)
        mz = pz + muzzle_h
        actual_yaw = float(self_data.get("aim", [0.0, 0.0])[0])

        best_target: Optional[EnemyTrack] = None
        best_score = -1e9
        best_ttk = 999.0
        best_imminent = False

        for e in active_enemies:
            ed = math.hypot(e.pos[0] - px, e.pos[1] - py)
            if ed > 245.0:
                continue

            has_los = self.world.has_line_of_sight(
                px, py, mz, e.pos[0], e.pos[1], e.pos[2] + 4.5
            )
            ttk = self.estimate_time_to_kill(self_data, e, t)

            # A kill is only imminent if we have clear line of sight right now
            # and lethal damage will actually land on the target within 1.15 seconds.
            kill_imminent = has_los and (ttk <= 1.15)

            score = 220.0 - min(20.0, ttk) * 15.0 - ed * 0.35
            if has_los:
                score += 120.0
            else:
                score -= 90.0

            if kill_imminent:
                score += 150.0
            elif has_los and ttk <= 2.2:
                score += 65.0

            eyaw = math.atan2(e.pos[1] - py, e.pos[0] - px)
            slew_rad = abs(angle_diff(actual_yaw, eyaw))
            score -= slew_rad * 12.0

            if primary_threat is not None and e.id == primary_threat.id:
                score += 18.0

            if (score > best_score + 1e-6) or (
                abs(score - best_score) <= 1e-6
                and best_target is not None
                and e.id < best_target.id
            ):
                best_score = score
                best_target = e
                best_ttk = ttk
                best_imminent = kill_imminent

        return best_target, best_ttk, best_imminent

    # ------------------------------------------------------------------
    # Upgrade Selection (#5: Nanites -> Speed -> Situational Laser / Armor)
    # ------------------------------------------------------------------

    def choose_upgrade(
        self,
        offer: Optional[Dict[str, Any]],
        self_data: Dict[str, Any],
        zone: Dict[str, Any],
        enemies: List[EnemyTrack],
    ) -> Optional[int]:
        if not offer or "choices" not in offer:
            return None
        choices = offer["choices"]
        if not choices:
            return None

        hp_frac = float(self_data.get("hp", 1400)) / max(
            1.0, float(self_data.get("max_hp", 1400))
        )
        zone_r = float(zone.get("radius", 560.0))
        next_r = float(zone.get("next_radius", 380.0))
        late_game = (zone_r <= 140.0) or (next_r <= 90.0)
        close_pressure = any(
            math.hypot(e.pos[0] - self_data["pos"][0], e.pos[1] - self_data["pos"][1]) < 60.0
            for e in enemies
        )

        has_laser = self.get_weapon_slot(self_data, "laser")[0] is not None
        has_lightning = self.get_weapon_slot(self_data, "lightning")[0] is not None

        best_idx = 0
        best_score = -1e9

        for idx, ch in enumerate(choices):
            cid = str(ch.get("id", "")).lower()
            cname = str(ch.get("name", "")).lower()
            cdesc = str(ch.get("desc", "")).lower()
            text = f"{cid} {cname} {cdesc}"

            score = 0.0

            if "nanite" in text or "regen" in text or "repair" in text:
                score += 100.0

            if "walk" in text or "speed" in text or "leg" in text or "servo" in text or "stride" in text:
                if "projectile" not in text and "shell" not in text and "bullet" not in text:
                    score += 80.0
            if "energy" in text or "capacitor" in text or "boost" in text or "reactor" in text:
                score += 65.0

            if "armor" in text or "damage taken" in text or "plating" in text:
                score += 85.0 if (late_game or close_pressure or hp_frac < 0.65) else 62.0
            if "max hp" in text or "hull" in text or "health" in text or "structure" in text:
                score += 82.0 if (late_game or close_pressure or hp_frac < 0.65) else 58.0

            if "laser" in text or "charge" in text or "beam" in text:
                if has_laser and not late_game and hp_frac >= 0.65:
                    score += 74.0
                elif has_laser:
                    score += 48.0
                else:
                    score += 15.0

            if "lightning" in text or "arc" in text or "chain" in text:
                if has_lightning or (late_game and self.preferred_secondary == "lightning"):
                    score += 76.0
                else:
                    score += 30.0

            if "gun" in text or "rotary" in text or "fire rate" in text or "ammo" in text or "reload" in text:
                score += 55.0

            if "sensor" in text or "vision" in text or "proximity" in text:
                score += 50.0
            if "turn" in text or "gyro" in text or "torso" in text:
                score += 45.0

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    # ------------------------------------------------------------------
    # Weapon Switching (#5: Encounter-driven switch with pickup & hysteresis)
    # ------------------------------------------------------------------

    def update_weapon_preference(
        self,
        obs: Dict[str, Any],
        self_data: Dict[str, Any],
        zone: Dict[str, Any],
        enemies: List[EnemyTrack],
        t: float,
    ) -> str:
        """
        Determines desired secondary weapon ("laser" vs "lightning") based on:
          - Actual close-range encounters (within lightning's ~45m effective envelope)
          - Opponents that favor lightning (e.g. laser carriers to short, or aggressive close brawlers)
          - Safely reachable pickup availability (when not already holding lightning)
          - Hysteresis cooldown (12s) so preference never oscillates rapidly.
        """
        # Respect hysteresis delay between preference flips
        if (t - self.last_pref_switch_t) < 12.0:
            return self.preferred_secondary

        px, py, pz = self_data["pos"]
        radius, height, _, _ = self.get_mech_dimensions(self_data)
        zone_r = float(zone.get("radius", 560.0))
        next_r = float(zone.get("next_radius", 380.0))

        has_lightning = self.get_weapon_slot(self_data, "lightning")[0] is not None
        has_laser = self.get_weapon_slot(self_data, "laser")[0] is not None

        # Check if there is a safely reachable lightning pickup when we don't hold one
        reachable_lightning_pickup = False
        for p in obs.get("pickups", []):
            if p.get("kind") != "lightning":
                continue
            ppos = p["pos"]
            if not self.is_inside_zone(ppos[0], ppos[1], zone, margin=8.0):
                continue
            euclid = math.hypot(ppos[0] - px, ppos[1] - py)
            if euclid > 65.0:
                continue
            rdist = self.world.reachable_distance(
                px,
                py,
                ppos[0],
                ppos[1],
                max_expansions=400,
                mech_radius=radius,
                mech_height=height,
                deadline_perf=self.tick_deadline_perf,
            )
            if rdist <= 75.0:
                reachable_lightning_pickup = True
                break

        # Evaluate opponent matchup & encounter distances
        close_enemies = [
            e for e in enemies if math.hypot(e.pos[0] - px, e.pos[1] - py) <= 48.0
        ]
        favorable_target_close = any(
            ("laser" in e.weapons or e.charging or math.hypot(e.pos[0] - px, e.pos[1] - py) <= 42.0)
            for e in close_enemies
        )
        # Caution against mortar carriers at medium range (mortar concusses & drains lightning energy)
        mortar_threat_mid = any(
            "mortar" in e.weapons and 46.0 < math.hypot(e.pos[0] - px, e.pos[1] - py) < 150.0
            for e in enemies
        )

        recent_window = self.recent_combat_dists[-5:]
        sustained_close_history = (
            len(recent_window) >= 3
            and (sum(recent_window) / len(recent_window)) <= 48.0
        )

        # Very final circle (radius <= 55m or next_radius <= 35m) forces everyone into ~45m
        very_tight_circle = (zone_r <= 55.0) or (next_r <= 35.0)
        compact_circle = (zone_r <= 105.0) or (next_r <= 55.0)

        want_lightning = False
        if has_lightning:
            # Keep lightning if encounters remain close or circle is compact
            if very_tight_circle or compact_circle or sustained_close_history or bool(close_enemies):
                want_lightning = True
        else:
            # To switch away from laser to lightning, require reachable pickup (or empty slot)
            # AND actual close-quarters evidence + favorable opponent matchup
            can_acquire = reachable_lightning_pickup or (not has_laser)
            if can_acquire and not mortar_threat_mid:
                if very_tight_circle and ( close_enemies or sustained_close_history ):
                    want_lightning = True
                elif compact_circle and favorable_target_close:
                    want_lightning = True
                elif sustained_close_history and favorable_target_close:
                    want_lightning = True

        new_pref = "lightning" if want_lightning else "laser"
        if new_pref != self.preferred_secondary:
            self.preferred_secondary = new_pref
            self.last_pref_switch_t = t
        return self.preferred_secondary

    def desired_secondary_kind(
        self,
        obs: Dict[str, Any],
        self_data: Dict[str, Any],
        zone: Dict[str, Any],
        enemies: List[EnemyTrack],
        t: float,
    ) -> str:
        return self.update_weapon_preference(obs, self_data, zone, enemies, t)

    def evaluate_weapon_pickups(
        self,
        obs: Dict[str, Any],
        self_data: Dict[str, Any],
        zone: Dict[str, Any],
        enemies: List[EnemyTrack],
        t: float,
    ) -> Tuple[Optional[int], Optional[int], Optional[Dict[str, Any]]]:
        """
        Returns (take_id, drop_slot, target_pickup_to_walk_to).
        Ensures we keep `gun` in one slot and our desired secondary in the other.
        """
        weapons = self_data.get("weapons", [None, None])
        held_kinds = [w.get("kind") if w else None for w in weapons]
        want_sec = self.desired_secondary_kind(obs, self_data, zone, enemies, t)

        if want_sec == "laser":
            sec_rank = {"laser": 3, "lightning": 2, "mortar": 1, None: 0}
        else:
            sec_rank = {"lightning": 3, "laser": 2, "mortar": 1, None: 0}

        gun_slot, _ = self.get_weapon_slot(self_data, "gun")
        if gun_slot is not None:
            sec_slot = 1 - gun_slot
            cur_sec = held_kinds[sec_slot]
        else:
            sec_slot = 1 if held_kinds[0] is not None else 0
            cur_sec = held_kinds[sec_slot]

        cur_rank = sec_rank.get(cur_sec, 0)

        sec_w = weapons[sec_slot] if sec_slot in (0, 1) else None
        sec_busy = bool(
            sec_w and sec_w.get("kind") == "laser" and float(sec_w.get("charge", 0.0)) > 0.0
        )

        pickups = obs.get("pickups", [])
        if not pickups:
            return None, None, None

        px, py, pz = self_data["pos"]
        radius, height, _, _ = self.get_mech_dimensions(self_data)
        loot_cfg = self.config.get("loot", {})
        pickup_r = float(loot_cfg.get("pickup_radius", 4.0))

        best_immediate: Optional[Tuple[int, Optional[int], int]] = None
        best_walk_pickup: Optional[Dict[str, Any]] = None
        best_walk_score = -1e9

        for p in pickups:
            pkind = p.get("kind")
            pid = int(p["id"])
            ppos = p["pos"]
            dx = ppos[0] - px
            dy = ppos[1] - py
            dz = abs(ppos[2] - pz)
            dist = math.hypot(dx, dy)

            if pkind in held_kinds:
                continue
            if self.is_goal_blacklisted(ppos[0], ppos[1], t):
                continue

            if gun_slot is None and pkind == "gun":
                rank_gain = 10
                target_slot = (
                    0 if held_kinds[0] is None else (1 if held_kinds[1] is None else sec_slot)
                )
            else:
                new_rank = sec_rank.get(pkind, 0)
                if new_rank <= cur_rank:
                    continue
                rank_gain = new_rank - cur_rank
                target_slot = sec_slot

            if dist <= pickup_r - 0.15 and dz <= 3.9:
                if held_kinds[target_slot] is None:
                    drop_slot = None
                    can_take = True
                else:
                    drop_slot = target_slot
                    can_take = not sec_busy
                if can_take:
                    if best_immediate is None or rank_gain > best_immediate[2]:
                        best_immediate = (pid, drop_slot, rank_gain)

            if dz <= 3.0 and self.is_inside_zone(ppos[0], ppos[1], zone, margin=12.0):
                if dist < 85.0 and self.is_loot_point_safe_from_squad(
                    ppos[0], ppos[1], ppos[2], self_data, enemies, max_safe_travel_m=42.0
                ):
                    rdist = self.world.reachable_distance(
                        px,
                        py,
                        ppos[0],
                        ppos[1],
                        max_expansions=450,
                        mech_radius=radius,
                        mech_height=height,
                        deadline_perf=self.tick_deadline_perf,
                    )
                    if rdist < 130.0:
                        walk_score = rank_gain * 45.0 - rdist
                        if walk_score > best_walk_score:
                            best_walk_score = walk_score
                            best_walk_pickup = p

        if best_immediate is not None:
            return best_immediate[0], best_immediate[1], best_walk_pickup
        return None, None, best_walk_pickup

    # ------------------------------------------------------------------
    # Zone, Cover Evaluation & Global Pathfinding (#3 & #6)
    # ------------------------------------------------------------------

    def is_inside_zone(
        self, x: float, y: float, zone: Dict[str, Any], margin: float = 0.0
    ) -> bool:
        cx, cy = zone.get("center", [0.0, 0.0])
        r = float(zone.get("radius", 560.0))
        return math.hypot(x - cx, y - cy) <= max(1.0, r - margin)

    def is_inside_next_zone(
        self, x: float, y: float, zone: Dict[str, Any], margin: float = 0.0
    ) -> bool:
        ncx, ncy = zone.get("next_center", zone.get("center", [0.0, 0.0]))
        nr = float(zone.get("next_radius", zone.get("radius", 380.0)))
        return math.hypot(x - ncx, y - ncy) <= max(1.0, nr - margin)

    def is_goal_blacklisted(self, gx: float, gy: float, t: float) -> bool:
        self.blacklisted_goals = [
            (bx, by, exp) for (bx, by, exp) in self.blacklisted_goals if exp > t
        ]
        return any(
            math.hypot(gx - bx, gy - by) < 10.0 for (bx, by, _) in self.blacklisted_goals
        )

    def is_loot_point_safe_from_squad(
        self,
        lx: float,
        ly: float,
        lz: float,
        self_data: Dict[str, Any],
        enemies: List[EnemyTrack],
        max_safe_travel_m: float = 48.0,
    ) -> bool:
        """
        Checks whether collecting a reward at (lx, ly, lz) is safe:
        - Rejects loot if any surviving enemy has clear line of sight to (lx, ly) within 165m
          or is closer to the loot than we are.
        - If enemies are tracked nearby, requires the loot to be within a quick dash
          (`max_safe_travel_m`) so we can grab it and return to cover before the squad closes.
        """
        px, py, _ = self_data["pos"]
        _, _, eye_h, _ = self.get_mech_dimensions(self_data)
        my_dist = math.hypot(lx - px, ly - py)

        if enemies and my_dist > max_safe_travel_m:
            return False

        for e in enemies:
            ed = math.hypot(lx - e.pos[0], ly - e.pos[1])
            if ed < 42.0 or (ed < my_dist + 8.0 and ed < 95.0):
                return False
            if ed < 165.0:
                if self.world.has_line_of_sight(
                    lx, ly, lz + eye_h * 0.75, e.pos[0], e.pos[1], e.pos[2] + 6.5
                ):
                    return False
        return True

    def select_cover_anchor_with_info(
        self,
        self_data: Dict[str, Any],
        zone: Dict[str, Any],
        enemies: List[EnemyTrack],
        t: float,
        force_break_los: bool = False,
    ) -> Tuple[Tuple[float, float], bool, float]:
        """
        Finds a tactical position inside the next safe circle with building/terrain
        cover blocking enemy sightlines, at least 2 connected escape routes, and
        scores candidates by reachable A* travel distance.
        Returns ((anchor_x, anchor_y), breaks_all_enemy_los, reachable_travel_dist).
        """
        px, py, pz = self_data["pos"]
        radius, height, eye_h, _ = self.get_mech_dimensions(self_data)

        cx, cy = zone.get("center", [0.0, 0.0])
        cur_r = float(zone.get("radius", 560.0))
        ncx, ncy = zone.get("next_center", [cx, cy])
        next_r = float(zone.get("next_radius", cur_r))

        # Keep existing anchor if still valid
        if (
            self.anchor_pos is not None
            and (t - self.anchor_refresh_t) < (1.2 if force_break_los else 5.0)
            and not self.is_goal_blacklisted(self.anchor_pos[0], self.anchor_pos[1], t)
        ):
            ax, ay = self.anchor_pos
            if self.is_inside_next_zone(ax, ay, zone, margin=8.0):
                az = self.world.ground_height(ax, ay)
                all_hid = all(
                    not self.world.has_line_of_sight(
                        ax, ay, az + eye_h, e.pos[0], e.pos[1], e.pos[2] + 6.5
                    )
                    for e in enemies
                )
                if (not force_break_los) or all_hid:
                    rdist = self.world.reachable_distance(
                        px,
                        py,
                        ax,
                        ay,
                        max_expansions=320,
                        mech_radius=radius,
                        mech_height=height,
                        deadline_perf=self.tick_deadline_perf,
                    )
                    if rdist < 180.0:
                        return self.anchor_pos, all_hid, rdist

        candidates: List[Tuple[float, float]] = []

        if enemies:
            tx = sum(e.pos[0] for e in enemies) / len(enemies)
            ty = sum(e.pos[1] for e in enemies) / len(enemies)
        elif self.last_damaged_bearing is not None and (t - self.last_damaged_t) < 5.0:
            tx = px + 60.0 * math.cos(self.last_damaged_bearing)
            ty = py + 60.0 * math.sin(self.last_damaged_bearing)
        else:
            tx, ty = ncx, ncy

        search_r = 130.0 if not force_break_los else 95.0
        if self.world.loaded and self.world.cover_buildings:
            for b in self.world.cover_buildings:
                dist_b = math.hypot(b.cx - px, b.cy - py)
                if dist_b > search_r:
                    continue
                if math.hypot(b.cx - cx, b.cy - cy) > cur_r - 10.0:
                    continue

                away_yaw = math.atan2(b.cy - ty, b.cx - tx)
                half_w = 0.5 * max(b.max_x - b.min_x, b.max_y - b.min_y) + radius + 4.0
                for ang_off in (0.0, 0.65, -0.65, 1.3, -1.3):
                    ang = away_yaw + ang_off
                    cand_x = b.cx + math.cos(ang) * half_w
                    cand_y = b.cy + math.sin(ang) * half_w
                    candidates.append((cand_x, cand_y))

        for ring_d in (24.0, 48.0, 75.0):
            for k in range(8):
                ang = k * (math.pi * 0.25)
                candidates.append((px + math.cos(ang) * ring_d, py + math.sin(ang) * ring_d))

        dir_from_next = math.atan2(py - ncy, px - ncx)
        sweet_r = min(next_r * 0.55, max(0.0, next_r - 18.0))
        candidates.append(
            (ncx + math.cos(dir_from_next) * sweet_r, ncy + math.sin(dir_from_next) * sweet_r)
        )

        # Pre-score candidates geometrically without running expensive BFS escapes on every candidate
        prelim: List[Tuple[float, float, float, bool]] = []

        for cand_x, cand_y in candidates:
            if self.is_goal_blacklisted(cand_x, cand_y, t):
                continue
            dist_cur = math.hypot(cand_x - cx, cand_y - cy)
            if dist_cur > cur_r - 8.0:
                continue

            cand_z = self.world.ground_height(cand_x, cand_y)
            if self.world.is_point_blocked(cand_x, cand_y, cand_z, radius, height):
                continue

            score = 0.0

            dist_next = math.hypot(cand_x - ncx, cand_y - ncy)
            if dist_next <= next_r - 8.0:
                score += 45.0
                if next_r > 40.0:
                    frac = dist_next / next_r
                    if 0.30 <= frac <= 0.78:
                        score += 20.0
            else:
                score -= (dist_next - next_r) * 1.2

            all_hidden = True
            exposed_count = 0
            for e in enemies:
                edist = math.hypot(cand_x - e.pos[0], cand_y - e.pos[1])
                has_los = self.world.has_line_of_sight(
                    cand_x,
                    cand_y,
                    cand_z + eye_h,
                    e.pos[0],
                    e.pos[1],
                    e.pos[2] + 6.5,
                )
                reach = self.get_enemy_weapon_reach(e)
                if has_los and edist <= reach:
                    all_hidden = False
                    exposed_count += 1
                if force_break_los:
                    score += 80.0 if not has_los else -65.0
                    score += min(edist, 140.0) * 0.45
                else:
                    if "lightning" in e.weapons and edist < 55.0:
                        score -= (55.0 - edist) * 2.5
                    elif 80.0 <= edist <= 150.0:
                        score += 25.0
                    elif edist < 50.0:
                        score -= (50.0 - edist) * 1.2
                    if not has_los:
                        score += 24.0

            # Duelist corner preference: strongly reward spots exposed to at most 1 enemy
            # and heavily penalize positions where 2+ shooters have simultaneous sightlines
            if len(enemies) >= 2:
                if exposed_count == 0:
                    score += 55.0
                elif exposed_count == 1 and not force_break_los:
                    score += 40.0
                elif exposed_count >= 2:
                    score -= 95.0 * (exposed_count - 1)

            euclid_d = math.hypot(cand_x - px, cand_y - py)
            prelim.append((score - euclid_d * 0.35, cand_x, cand_y, all_hidden))

        prelim.sort(key=lambda item: item[0], reverse=True)

        best_pt = (ncx, ncy)
        best_score = -1e9
        best_hidden = False
        best_reach_d = math.hypot(ncx - px, ncy - py)

        # Evaluate top candidates for connected escape routes + reachable A* travel distance
        evaluated = 0
        for base_s, cand_x, cand_y, all_hidden in prelim:
            if evaluated >= 3:
                break
            cand_z = self.world.ground_height(cand_x, cand_y)
            escapes = self.world.open_escape_directions(
                cand_x, cand_y, cand_z, probe_dist=20.0, mech_radius=radius, mech_height=height
            )
            if escapes < 2:
                continue
            evaluated += 1
            escape_bonus = 25.0 if (2 <= escapes <= 6) else 10.0
            euclid_d = math.hypot(cand_x - px, cand_y - py)
            reach_d = self.world.reachable_distance(
                px,
                py,
                cand_x,
                cand_y,
                max_expansions=260,
                mech_radius=radius,
                mech_height=height,
                deadline_perf=self.tick_deadline_perf,
            )
            if reach_d > 300.0:
                continue
            true_score = (base_s + euclid_d * 0.35 + escape_bonus) - reach_d * 0.55
            if true_score > best_score:
                best_score = true_score
                best_pt = (cand_x, cand_y)
                best_hidden = all_hidden
                best_reach_d = reach_d

        self.anchor_pos = best_pt
        self.anchor_refresh_t = t
        return best_pt, best_hidden, best_reach_d

    def select_cover_anchor(
        self,
        self_data: Dict[str, Any],
        zone: Dict[str, Any],
        enemies: List[EnemyTrack],
        t: float,
        force_break_los: bool = False,
    ) -> Tuple[float, float]:
        pt, _, _ = self.select_cover_anchor_with_info(
            self_data, zone, enemies, t, force_break_los=force_break_los
        )
        return pt

    def compute_navigation_move(
        self,
        self_data: Dict[str, Any],
        goal_x: float,
        goal_y: float,
        t: float,
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """
        Combines global A* pathfinding with local obstacle steering and automatic
        stuck detection/recovery.
        """
        px, py, pz = self_data["pos"]
        radius, height, _, _ = self.get_mech_dimensions(self_data)
        dist_to_goal = math.hypot(goal_x - px, goal_y - py)
        if dist_to_goal <= 1.2:
            self.current_path = []
            return [0.0, 0.0], []

        # 1. Active stuck recovery maneuver if recently trapped against geometry
        if t < self.stuck_until_t:
            if (
                self.path_goal is not None
                and math.hypot(goal_x - self.path_goal[0], goal_y - self.path_goal[1]) <= 8.0
            ):
                return [round(self.unstuck_dir[0], 4), round(self.unstuck_dir[1], 4)], self.current_path
            self.stuck_until_t = -100.0

        # 2. Check progress over recent 1.2s window when commanding full movement
        if len(self.progress_history) >= 8:
            old_t, old_x, old_y, _ = self.progress_history[0]
            dt = t - old_t
            avg_cmd = sum(item[3] for item in self.progress_history) / len(
                self.progress_history
            )
            net_disp = math.hypot(px - old_x, py - old_y)
            if dt >= 0.75 and avg_cmd >= 0.7 and net_disp < 1.35:
                # We are stuck! Blacklist current immediate goal/anchor, flip lateral escape direction, and replan
                self.blacklisted_goals.append((goal_x, goal_y, t + 8.0))
                self.anchor_pos = None
                self.path_goal = (goal_x, goal_y)
                self.current_path = []
                self.progress_history.clear()

                # Pick the clearest lateral/reverse escape direction
                base_ang = math.atan2(goal_y - py, goal_x - px)
                self.unstuck_side = -self.unstuck_side
                best_ang = base_ang + self.unstuck_side * (math.pi * 0.65)
                best_clear = -1.0
                for cand_off in (
                    self.unstuck_side * 1.35,
                    -self.unstuck_side * 1.35,
                    self.unstuck_side * 2.1,
                    -self.unstuck_side * 2.1,
                    math.pi,
                ):
                    ang = base_ang + cand_off
                    tx = px + math.cos(ang) * 12.0
                    ty = py + math.sin(ang) * 12.0
                    if self.world.is_walkable_segment_2d(px, py, tx, ty, radius, height):
                        best_ang = ang
                        best_clear = 12.0
                        break
                self.unstuck_dir = (math.cos(best_ang), math.sin(best_ang))
                self.stuck_until_t = t + 0.65
                return [round(self.unstuck_dir[0], 4), round(self.unstuck_dir[1], 4)], []

        # 3. Replan A* path when goal changes or periodic refresh expires
        need_replan = (
            self.path_goal is None
            or math.hypot(goal_x - self.path_goal[0], goal_y - self.path_goal[1]) > 6.0
            or not self.current_path
            or (t - self.path_refresh_t) > 1.5
        )
        if need_replan:
            waypoints, _ = self.world.find_path_astar(
                px,
                py,
                goal_x,
                goal_y,
                max_expansions=1600,
                mech_radius=radius,
                mech_height=height,
                smooth=True,
                deadline_perf=self.tick_deadline_perf,
            )
            self.current_path = waypoints
            self.path_goal = (goal_x, goal_y)
            self.path_refresh_t = t

        # Advance past reached waypoints
        while len(self.current_path) > 1:
            wx, wy = self.current_path[0]
            if math.hypot(wx - px, wy - py) < 4.2:
                self.current_path.pop(0)
            else:
                break

        wp_x, wp_y = self.current_path[0] if self.current_path else (goal_x, goal_y)
        mv_x, mv_y = self.world.steer_around_obstacles(
            self_data["pos"], wp_x, wp_y, radius, height
        )
        speed_scale = 1.0 if dist_to_goal > 4.0 else (dist_to_goal / 4.0)
        cmd = [round(mv_x * speed_scale, 4), round(mv_y * speed_scale, 4)]

        cmd_mag = math.hypot(cmd[0], cmd[1])
        self.progress_history.append((t, px, py, cmd_mag))
        if len(self.progress_history) > 12:
            self.progress_history.pop(0)

        return cmd, self.current_path

    # ------------------------------------------------------------------
    # Laser Discharge & Aim Coordination (#2: Explicit laser states)
    # ------------------------------------------------------------------

    def find_safe_laser_discharge_aim(
        self,
        self_data: Dict[str, Any],
        enemies: List[EnemyTrack],
        fallback_yaw: float,
    ) -> Tuple[float, float]:
        """
        Finds a (yaw, pitch) direction that is guaranteed not to hit any known
        enemy mech, preferring a nearby wall/ground impact or high-pitch sky shot
        while keeping the enemy inside our 180-deg front armor arc if possible.
        """
        px, py, pz = self_data["pos"]
        _, _, _, muzzle_h = self.get_mech_dimensions(self_data)
        mz = pz + muzzle_h
        actual_yaw = float(self_data.get("aim", [0.0, 0.0])[0])

        enemy_bearings: List[Tuple[float, float]] = []
        for e in enemies:
            ed = math.hypot(e.pos[0] - px, e.pos[1] - py)
            eyaw = math.atan2(e.pos[1] - py, e.pos[0] - px)
            safe_cone = max(0.22, math.atan2(6.0, max(4.0, ed)))
            enemy_bearings.append((eyaw, safe_cone))

        def is_yaw_clear_of_enemies(cand_yaw: float) -> bool:
            return all(
                abs(angle_diff(cand_yaw, eyaw)) > cone
                for eyaw, cone in enemy_bearings
            )

        # First check if our current actual yaw is already safe to discharge immediately
        if is_yaw_clear_of_enemies(actual_yaw):
            return actual_yaw, -0.55

        # Otherwise pick the closest yaw offset from actual_yaw that clears all enemies
        for off in (0.35, -0.35, 0.65, -0.65, 1.0, -1.0, 1.4, -1.4):
            cand = wrap_angle(actual_yaw + off)
            if is_yaw_clear_of_enemies(cand):
                return cand, -0.55

        return wrap_angle(fallback_yaw + 0.8), -0.65

    # ------------------------------------------------------------------
    # Supply Drops (#6: Capture when plausibly uncontested)
    # ------------------------------------------------------------------

    def evaluate_supply_drop(
        self,
        obs: Dict[str, Any],
        self_data: Dict[str, Any],
        zone: Dict[str, Any],
        enemies: List[EnemyTrack],
    ) -> Optional[Dict[str, Any]]:
        drops = obs.get("drops", [])
        if not drops:
            return None

        px, py, _ = self_data["pos"]
        radius, height, _, _ = self.get_mech_dimensions(self_data)
        my_id = int(self_data.get("id", self.slot))
        hp_frac = float(self_data.get("hp", 1400)) / max(
            1.0, float(self_data.get("max_hp", 1400))
        )

        best_drop: Optional[Dict[str, Any]] = None
        best_score = -1e9

        for d in drops:
            dpos = d["pos"]
            lands_in = float(d.get("lands_in", 0.0))
            holder = d.get("holder")
            dist = math.hypot(dpos[0] - px, dpos[1] - py)

            if not self.is_inside_zone(dpos[0], dpos[1], zone, margin=6.0):
                continue
            if lands_in > 8.0 and dist > 60.0:
                continue
            if lands_in > 14.0:
                continue

            contested = False
            for e in enemies:
                edist = math.hypot(e.pos[0] - dpos[0], e.pos[1] - dpos[1])
                if edist < 55.0 or (edist < dist - 10.0 and edist < 105.0):
                    contested = True
                    break
                # Skip supply drop if an enemy has a clear firing lane onto the ring
                if edist < 165.0 and self.world.has_line_of_sight(
                    dpos[0], dpos[1], dpos[2] + 6.0, e.pos[0], e.pos[1], e.pos[2] + 6.5
                ):
                    contested = True
                    break
            if holder is not None and holder != my_id:
                contested = True

            if contested:
                continue

            rdist = self.world.reachable_distance(
                px,
                py,
                dpos[0],
                dpos[1],
                max_expansions=500,
                mech_radius=radius,
                mech_height=height,
                deadline_perf=self.tick_deadline_perf,
            )
            if rdist > 140.0:
                continue

            heal_bonus = (1.0 - hp_frac) * 90.0
            progress_bonus = float(d.get("progress", 0.0)) * 25.0 if holder == my_id else 0.0
            score = 110.0 + heal_bonus + progress_bonus - rdist - lands_in * 4.0

            if score > best_score:
                best_score = score
                best_drop = d

        return best_drop

    # ------------------------------------------------------------------
    # Early/Mid Scavenging (#1: Cubes, Isolated Crates & Drones)
    # ------------------------------------------------------------------

    def find_farming_target(
        self,
        obs: Dict[str, Any],
        self_data: Dict[str, Any],
        zone: Dict[str, Any],
        enemies: List[EnemyTrack],
        t: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Returns (best_ground_cube_to_collect, best_crate_or_npc_to_farm).
        Only farms when there is a safe reachable collection route away from enemy mechs.
        """
        px, py, pz = self_data["pos"]
        radius, height, _, _ = self.get_mech_dimensions(self_data)
        my_cubes = int(self_data.get("cubes", 0))

        best_cube: Optional[Dict[str, Any]] = None
        best_cube_score = -1e9
        for c in obs.get("cubes", []):
            cpos = c["pos"]
            if self.is_goal_blacklisted(cpos[0], cpos[1], t):
                continue
            if not self.is_inside_zone(cpos[0], cpos[1], zone, margin=5.0):
                continue
            if (cpos[2] - pz) > 2.5:
                continue
            dist = math.hypot(cpos[0] - px, cpos[1] - py)
            if dist > 90.0:
                continue

            # Only collect cubes when safe from surviving enemy firing lanes
            if not self.is_loot_point_safe_from_squad(
                cpos[0], cpos[1], cpos[2], self_data, enemies, max_safe_travel_m=38.0
            ):
                continue

            rdist = self.world.reachable_distance(
                px,
                py,
                cpos[0],
                cpos[1],
                max_expansions=400,
                mech_radius=radius,
                mech_height=height,
                deadline_perf=self.tick_deadline_perf,
            )
            if rdist > 120.0:
                continue

            score = 100.0 - rdist
            if score > best_cube_score:
                best_cube_score = score
                best_cube = c

        if my_cubes >= 12 and enemies:
            return best_cube, None

        best_farm: Optional[Dict[str, Any]] = None
        best_farm_score = -1e9

        for cr in obs.get("crates", []):
            cpos = cr["pos"]
            if not self.is_inside_zone(cpos[0], cpos[1], zone, margin=12.0):
                continue
            if (cpos[2] - pz) > 3.5:
                continue
            dist = math.hypot(cpos[0] - px, cpos[1] - py)
            if dist > 110.0:
                continue
            if any(
                math.hypot(cpos[0] - e.pos[0], cpos[1] - e.pos[1]) < 65.0
                for e in enemies
            ):
                continue

            hp = float(cr.get("hp", 300.0))
            score = 80.0 - dist * 0.6 + (300.0 - hp) * 0.1
            if score > best_farm_score:
                best_farm_score = score
                best_farm = {
                    "id": cr["id"],
                    "kind": "crate",
                    "pos": cpos,
                    "vel": [0.0, 0.0, 0.0],
                    "hp": hp,
                    "aim_z_offset": 1.2,
                }

        if not enemies and float(self_data.get("hp", 1400)) > 950:
            for m in obs.get("mechs", []):
                if m.get("kind") != "npc":
                    continue
                npos = m["pos"]
                if not self.is_inside_zone(npos[0], npos[1], zone, margin=15.0):
                    continue
                dist = math.hypot(npos[0] - px, npos[1] - py)
                if dist > 95.0:
                    continue
                hp = float(m.get("hp", 350.0))
                score = 72.0 - dist * 0.6 + (350.0 - hp) * 0.12
                if score > best_farm_score:
                    best_farm_score = score
                    best_farm = {
                        "id": m["id"],
                        "kind": "npc",
                        "pos": npos,
                        "vel": m.get("vel", [0.0, 0.0, 0.0]),
                        "hp": hp,
                        "aim_z_offset": 1.75,
                    }

        return best_cube, best_farm

    # ------------------------------------------------------------------
    # Main Decision Loop (50ms budget)
    # ------------------------------------------------------------------

    def on_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Set a shared planning deadline (~28ms from start of tick) so A* queries never breach the 50ms budget
        budget_ms = float(self.config.get("bot", {}).get("tick_budget_ms", 50.0))
        self.tick_deadline_perf = time.perf_counter() + min(0.028, budget_ms * 0.00056)

        t = float(obs.get("t", 0.0))
        self_data = obs["self"]
        zone = obs.get("zone", {})
        px, py, pz = self_data["pos"]
        hp = float(self_data.get("hp", 1400.0))
        max_hp = max(1.0, float(self_data.get("max_hp", 1400.0)))
        hp_frac = hp / max_hp
        energy = float(self_data.get("energy", 100.0))
        my_cubes = int(self_data.get("cubes", 0))
        actual_yaw, actual_pitch = self_data.get("aim", [0.0, 0.0])
        in_zone = bool(self_data.get("in_zone", True))

        radius, height, eye_h, muzzle_h = self.get_mech_dimensions(self_data)
        dbg = DebugDraw(self.config.get("debug"))

        # 1. Process events: track damage received, hits dealt, and confirmed kills.
        # Note: Only a confirmed `"kill"` event (or current `obs.mechs` showing hp <= 0)
        # marks a mech dead, because an opponent can pick up a cube or supply drop
        # on the same tick our hit lands!
        confirmed_kill_ids = set()
        hit_damage_by_target: Dict[int, float] = {}
        for ev in obs.get("events", []):
            etype = ev.get("type")
            if etype == "damaged":
                self.last_damaged_t = t
                self.last_combat_t = t
                if ev.get("bearing") is not None:
                    self.last_damaged_bearing = float(ev["bearing"])
                if ev.get("by") is not None:
                    self.last_damaged_by = int(ev["by"])
            elif etype == "hit":
                # Dealing damage to any target also resets the server's regeneration timer
                self.last_dealt_damage_t = t
                self.last_combat_t = t
                target_id = ev.get("target")
                dmg_amount = float(ev.get("amount", 0.0))
                if target_id is not None:
                    tid = int(target_id)
                    hit_damage_by_target[tid] = hit_damage_by_target.get(tid, 0.0) + dmg_amount
            elif etype == "kill":
                target_id = ev.get("target")
                if target_id is not None:
                    confirmed_kill_ids.add(int(target_id))
            elif etype == "effect":
                if ev.get("effect") == "short":
                    short_s = float(
                        self.config.get("weapons", {})
                        .get("lightning", {})
                        .get("short_s", 2.0)
                    )
                    self.shorted_until_t = t + short_s

        for kid in confirmed_kill_ids:
            self.enemy_tracks.pop(kid, None)
            if self.last_damaged_by == kid:
                self.last_damaged_by = None

        if not in_zone:
            self.last_damaged_t = t
            self.last_combat_t = t

        last_regen_reset_t = max(self.last_damaged_t, self.last_dealt_damage_t)
        regen_delay = self.get_regen_delay(self_data)
        time_since_combat = t - last_regen_reset_t
        time_until_regen = max(0.0, regen_delay - time_since_combat)
        is_regenerating = in_zone and (time_since_combat >= regen_delay) and (hp < max_hp)

        # 2. Update enemy tracks (vision memory = 5s) & sound pings.
        # Fresh observation HP always overrides any prior damage estimate.
        visible_enemy_ids = set()
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech":
                mid = int(m["id"])
                if mid in confirmed_kill_ids or float(m.get("hp", 1400.0)) <= 0.0:
                    self.enemy_tracks.pop(mid, None)
                    continue
                tr = EnemyTrack(m, t)
                self.enemy_tracks[tr.id] = tr
                visible_enemy_ids.add(tr.id)

        # For enemies remembered from earlier ticks (not in current obs.mechs) that we hit
        # blind without a kill event, clamp their estimated HP to >= 1.0 so we never
        # falsely discard a live opponent without a confirmed kill event.
        for tid, dmg in hit_damage_by_target.items():
            if tid not in visible_enemy_ids and tid not in confirmed_kill_ids:
                tr = self.enemy_tracks.get(tid)
                if tr is not None:
                    tr.hp = max(1.0, tr.hp - dmg)

        memory_s = float(self.config.get("vision", {}).get("memory_s", 5.0))
        stale_ids = [
            eid
            for eid, tr in self.enemy_tracks.items()
            if (t - tr.last_seen_t) > memory_s + 1.5
        ]
        for eid in stale_ids:
            del self.enemy_tracks[eid]

        active_enemies = list(self.enemy_tracks.values())

        for snd in obs.get("sounds", []):
            skind = str(snd.get("kind", ""))
            if skind in (
                "gun",
                "laser",
                "laser_charge",
                "lightning",
                "mortar",
                "step",
                "boost",
                "jet",
            ):
                sbearing = float(snd.get("bearing", 0.0))
                sdist = str(snd.get("dist", "mid"))
                sage = float(snd.get("age", 0.0))
                self.sound_pings.append(SoundPing(skind, sbearing, sdist, t - sage))
        self.sound_pings = [p for p in self.sound_pings if (t - p.t) < 3.5]

        # Record encounter distances when enemies are genuinely engaged (< 120m)
        for e in active_enemies:
            if e.id in visible_enemy_ids:
                ed = math.hypot(e.pos[0] - px, e.pos[1] - py)
                if ed < 120.0:
                    self.recent_combat_dists.append(ed)
                    if len(self.recent_combat_dists) > 16:
                        self.recent_combat_dists.pop(0)

        # 3. Identify primary enemy threat (for defense/cover), all active shooters, AND best firing target
        primary_enemy, primary_enemy_dist, shooters = (
            self.select_primary_threat(self_data, active_enemies, t)
        )
        threatening_enemies_count = len(shooters)
        if self.enable_finishable_targeting:
            firing_enemy, firing_ttk, kill_imminent = self.select_firing_target(
                self_data, active_enemies, primary_enemy, t
            )
        else:
            firing_enemy, firing_ttk, kill_imminent = primary_enemy, 99.0, False

        # 4. Evaluate Reposition / Retreat state (#2: Reposition immediately on 2+ shooters unless kill is imminent)
        stronger_closing = False
        if primary_enemy is not None:
            enemy_power = primary_enemy.hp * (1.0 + 0.04 * min(10, primary_enemy.cubes))
            my_power = hp * (1.0 + 0.04 * min(10, my_cubes))
            has_close_counter = (
                "lightning" in primary_enemy.weapons and primary_enemy_dist < 58.0
            )
            if (enemy_power > my_power * 1.25 and primary_enemy_dist < 75.0) or (
                has_close_counter and self.get_weapon_slot(self_data, "lightning")[0] is None
            ):
                stronger_closing = True

        # Reposition as soon as a second enemy can shoot us (threatening_enemies_count >= 2)
        # even at 100% HP, UNLESS a kill on our firing target is imminent and our HP is healthy enough to finish it.
        if self.enable_early_multi_reposition:
            multi_shooter_pressure = threatening_enemies_count >= 2 and not (
                kill_imminent and hp_frac >= 0.45
            )
        else:
            multi_shooter_pressure = threatening_enemies_count >= 2 and hp_frac < 0.85
        low_hp_retreat = (
            hp_frac < 0.62
            and (active_enemies or (t - self.last_damaged_t) < 4.0)
            and not (kill_imminent and hp_frac >= 0.38 and threatening_enemies_count <= 1)
        )

        should_enter_retreat = (
            low_hp_retreat
            or multi_shooter_pressure
            or (stronger_closing and hp_frac < 0.82 and not kill_imminent)
        )

        if self.mode == "RETREAT":
            # Exit retreat once we broke multi-shooter LOS and are healthy, or healed above 84%
            if (
                threatening_enemies_count <= 1
                and hp_frac >= 0.84
            ) or (
                hp_frac >= 0.72
                and not active_enemies
                and time_since_combat > regen_delay + 2.5
            ):
                self.mode = "SCAVENGE"
        elif should_enter_retreat:
            self.mode = "RETREAT"

        # 5. Evaluate Weapons, Upgrades, Supply Drops, and Scavenging targets
        pick_choice = self.choose_upgrade(
            obs.get("offer"), self_data, zone, active_enemies
        )
        take_id, drop_slot, walk_pickup = self.evaluate_weapon_pickups(
            obs, self_data, zone, active_enemies, t
        )
        target_drop = self.evaluate_supply_drop(obs, self_data, zone, active_enemies)
        best_cube, best_farm = self.find_farming_target(
            obs, self_data, zone, active_enemies, t
        )

        if self.mode != "RETREAT":
            if target_drop is not None:
                self.mode = "DROP"
            elif primary_enemy is not None and primary_enemy_dist < 175.0:
                self.mode = "COMBAT"
            else:
                self.mode = "SCAVENGE"

        # 6. Determine Retreat Substate & Movement Goal
        move_goal_x, move_goal_y = px, py
        want_boost = False
        want_jump = False

        cx, cy = zone.get("center", [0.0, 0.0])
        cur_r = float(zone.get("radius", 560.0))
        dist_to_center = math.hypot(px - cx, py - cy)
        urgent_zone = (not in_zone) or (dist_to_center > cur_r - 12.0)

        # Check if ANY enemy shooter currently has a clear firing lane within their weapon reach
        exposed_to_enemy = len(shooters) > 0
        closest_shooter_dist = (
            min(math.hypot(s.pos[0] - px, s.pos[1] - py) for s in shooters)
            if shooters
            else 1e9
        )

        if urgent_zone:
            ncx, ncy = zone.get("next_center", [cx, cy])
            move_goal_x, move_goal_y = ncx, ncy
            self.retreat_substate = "ESCAPING"
        elif self.mode == "RETREAT":
            cov_pt, cov_breaks_los, cov_reach_d = self.select_cover_anchor_with_info(
                self_data, zone, active_enemies, t, force_break_los=True
            )
            move_goal_x, move_goal_y = cov_pt

            # Classify retreat into HEALING, ESCAPING, or CORNERED using the full `shooters` set:
            # - HEALING: we are behind cover (no shooter has LOS) or already regenerating / within 1.25s of regen
            # - CORNERED: we are exposed to at least one shooter and cannot break LOS (no LOS-breaking cover reachable,
            #   or a shooter is right on top of us < 28m, or we took damage in the last 0.85s while cover is > 32m away)
            # - ESCAPING: exposed to shooter(s), moving toward reachable LOS-breaking cover
            recently_hit = (t - self.last_damaged_t) < 0.85
            can_reach_cover_soon = cov_breaks_los and (cov_reach_d <= 36.0 or energy >= 35.0)
            enemy_pressing_close = exposed_to_enemy and (closest_shooter_dist < 28.0)

            if not exposed_to_enemy:
                self.retreat_substate = "HEALING"
            elif enemy_pressing_close or not can_reach_cover_soon or (
                recently_hit and cov_reach_d > 32.0 and closest_shooter_dist < 80.0
            ):
                self.retreat_substate = "CORNERED"
            elif is_regenerating or (not recently_hit and time_until_regen <= 1.25):
                self.retreat_substate = "HEALING"
            else:
                self.retreat_substate = "ESCAPING"

        elif self.mode == "DROP" and target_drop is not None:
            dpos = target_drop["pos"]
            move_goal_x, move_goal_y = dpos[0], dpos[1]
        elif self.mode == "COMBAT" and primary_enemy is not None:
            ex, ey, ez = primary_enemy.pos
            has_lightning = self.get_weapon_slot(self_data, "lightning")[0] is not None
            enemy_has_lightning = "lightning" in primary_enemy.weapons

            if has_lightning:
                pref_min, pref_max = 20.0, 39.0
            elif enemy_has_lightning:
                pref_min, pref_max = 90.0, 150.0
            else:
                pref_min, pref_max = 80.0, 145.0

            laser_slot, laser_w = self.get_weapon_slot(self_data, "laser")
            laser_charge = float(laser_w.get("charge", 0.0)) if laser_w else 0.0
            laser_full = (
                float(laser_w.get("stats", {}).get("charge", 0.75)) if laser_w else 0.75
            )

            if primary_enemy.charging:
                cov_x, cov_y = self.select_cover_anchor(
                    self_data, zone, active_enemies, t, force_break_los=True
                )
                move_goal_x, move_goal_y = cov_x, cov_y
            elif laser_w and 0.0 < laser_charge < laser_full * 0.78:
                cov_x, cov_y = self.select_cover_anchor(
                    self_data, zone, active_enemies, t, force_break_los=True
                )
                move_goal_x, move_goal_y = cov_x, cov_y
            elif primary_enemy_dist < pref_min:
                away_yaw = math.atan2(py - ey, px - ex)
                kite_x = px + math.cos(away_yaw) * 35.0
                kite_y = py + math.sin(away_yaw) * 35.0
                if self.is_inside_zone(kite_x, kite_y, zone, margin=12.0) and not self.world.is_point_blocked(
                    kite_x, kite_y, None, radius, height
                ):
                    move_goal_x, move_goal_y = kite_x, kite_y
                else:
                    move_goal_x, move_goal_y = self.select_cover_anchor(
                        self_data, zone, active_enemies, t, force_break_los=False
                    )
            elif primary_enemy_dist > pref_max:
                to_yaw = math.atan2(ey - py, ex - px)
                move_goal_x = px + math.cos(to_yaw) * 28.0
                move_goal_y = py + math.sin(to_yaw) * 28.0
            else:
                cov_x, cov_y = self.select_cover_anchor(
                    self_data, zone, active_enemies, t, force_break_los=False
                )
                if math.hypot(cov_x - px, cov_y - py) > 8.0:
                    move_goal_x, move_goal_y = cov_x, cov_y
                else:
                    perp_yaw = math.atan2(ey - py, ex - px) + (
                        math.pi * 0.5 if (int(t * 0.7) % 2 == 0) else -math.pi * 0.5
                    )
                    move_goal_x = px + math.cos(perp_yaw) * 14.0
                    move_goal_y = py + math.sin(perp_yaw) * 14.0
        else:
            if walk_pickup is not None and (
                best_cube is None
                or math.hypot(walk_pickup["pos"][0] - px, walk_pickup["pos"][1] - py)
                < math.hypot(best_cube["pos"][0] - px, best_cube["pos"][1] - py) + 12.0
            ):
                move_goal_x, move_goal_y = walk_pickup["pos"][0], walk_pickup["pos"][1]
            elif best_cube is not None:
                move_goal_x, move_goal_y = best_cube["pos"][0], best_cube["pos"][1]
            elif (
                best_farm is not None
                and math.hypot(best_farm["pos"][0] - px, best_farm["pos"][1] - py) > 42.0
            ):
                move_goal_x, move_goal_y = best_farm["pos"][0], best_farm["pos"][1]
            else:
                move_goal_x, move_goal_y = self.select_cover_anchor(
                    self_data, zone, active_enemies, t, force_break_los=False
                )

        # Compute movement vector via A* pathfinding + local steering + stuck recovery
        move_cmd, active_path = self.compute_navigation_move(
            self_data, move_goal_x, move_goal_y, t
        )
        dist_to_goal = math.hypot(move_goal_x - px, move_goal_y - py)

        # Energy management
        has_lightning_equipped = (
            self.get_weapon_slot(self_data, "lightning")[0] is not None
        )
        energy_reserve = 55.0 if has_lightning_equipped else 45.0

        if urgent_zone and energy > 20.0:
            want_boost = True
        elif self.mode == "RETREAT":
            if (exposed_to_enemy or (t - self.last_damaged_t) < 1.2) and energy > 20.0:
                want_boost = True
        elif (
            primary_enemy is not None
            and (
                primary_enemy.charging
                or ("lightning" in primary_enemy.weapons and primary_enemy_dist < 50.0)
            )
            and energy > 30.0
        ):
            want_boost = True
        elif dist_to_goal > 65.0 and energy > max(78.0, energy_reserve + 25.0):
            want_boost = True

        # 7. Aiming, Firing & Laser State Machine (#2 & #4)
        fire_cmd = [False, False]
        want_reload = False
        desired_yaw = actual_yaw
        desired_pitch = 0.0
        self.laser_state = "IDLE"

        # Determine whether offensive fire is permitted:
        # - In COMBAT / SCAVENGE / DROP: permitted
        # - In RETREAT:
        #   * If we are repositioning at healthy HP (>= 78%) purely to break a 2nd shooter's sightline,
        #     or if a kill is imminent, we can take shots while moving to cover if already regenerated.
        #   * HEALING / ESCAPING (damaged < 78% HP and waiting on / running regeneration): cease fire strictly
        #   * CORNERED (exposed and unable to escape without taking fire): fight back and focus the most finishable opponent!
        allow_offensive_fire = True
        if self.mode == "RETREAT":
            if self.retreat_substate == "CORNERED":
                allow_offensive_fire = True
            elif kill_imminent and hp_frac >= 0.55:
                allow_offensive_fire = True
            elif hp_frac >= 0.96 and self.retreat_substate == "ESCAPING":
                allow_offensive_fire = True
            else:
                allow_offensive_fire = False

        # Separate defensive threat (`primary_enemy`, used for cover/positioning)
        # from offensive target (`firing_enemy`, the opponent we can finish fastest).
        target_enemy = firing_enemy if firing_enemy is not None else primary_enemy
        combat_target: Optional[Dict[str, Any]] = None
        if target_enemy is not None and allow_offensive_fire:
            combat_target = {
                "id": target_enemy.id,
                "kind": "mech",
                "pos": target_enemy.pos,
                "vel": target_enemy.vel,
                "hp": target_enemy.hp,
                "weapons": target_enemy.weapons,
                "aim_z_offset": 4.5,
            }
        elif self.mode == "SCAVENGE" and best_farm is not None and allow_offensive_fire:
            combat_target = best_farm

        gun_slot, gun_w = self.get_weapon_slot(self_data, "gun")
        laser_slot, laser_w = self.get_weapon_slot(self_data, "laser")
        light_slot, light_w = self.get_weapon_slot(self_data, "lightning")
        mortar_slot, mortar_w = self.get_weapon_slot(self_data, "mortar")

        muzzle_pos = (px, py, pz + muzzle_h)
        l_cur_charge = float(laser_w.get("charge", 0.0)) if laser_w else 0.0

        if combat_target is not None:
            tpos = combat_target["pos"]
            tvel = combat_target.get("vel", [0.0, 0.0, 0.0])
            tz_off = float(combat_target.get("aim_z_offset", 4.5))
            target_center = (tpos[0], tpos[1], tpos[2] + tz_off)
            tdist = math.hypot(tpos[0] - px, tpos[1] - py)

            has_los_to_target = self.world.has_line_of_sight(
                muzzle_pos[0],
                muzzle_pos[1],
                muzzle_pos[2],
                target_center[0],
                target_center[1],
                target_center[2],
            )

            hitscan_yaw, hitscan_pitch = yaw_pitch_to(
                muzzle_pos[0],
                muzzle_pos[1],
                muzzle_pos[2],
                target_center[0],
                target_center[1],
                target_center[2],
            )

            gun_speed = (
                float(gun_w.get("stats", {}).get("speed", 252.0)) if gun_w else 252.0
            )
            lead_x, lead_y, lead_z, _ = solve_intercept_3d(
                muzzle_pos,
                target_center,
                (tvel[0], tvel[1], tvel[2]),
                gun_speed,
                shot_origin_offset=radius + 0.5,
            )
            gun_yaw, gun_pitch = yaw_pitch_to(
                muzzle_pos[0], muzzle_pos[1], muzzle_pos[2], lead_x, lead_y, lead_z
            )

            # Target angular tolerances for torso alignment
            target_rad = 2.4 if combat_target["kind"] == "mech" else 1.4
            hitscan_yaw_tol = max(0.018, math.atan2(target_rad * 0.85, max(5.0, tdist)))
            hitscan_pitch_tol = max(0.025, math.atan2(2.8, max(5.0, tdist)))
            gun_yaw_tol = max(0.03, math.atan2(target_rad, max(5.0, tdist)))
            gun_pitch_tol = max(0.04, math.atan2(3.2, max(5.0, tdist)))

            # Coordinate Gun lead and Laser hitscan aim on our single torso:
            # - While laser is uncharged or early-charging (< 70%), aim at gun lead so the arm gun hits moving targets
            # - Once laser is >= 70% charged (or ready to release), slew torso to hitscan aim so the 315-dmg beam connects
            desired_yaw, desired_pitch = gun_yaw, gun_pitch

            if laser_slot is not None and laser_w is not None:
                l_stats = laser_w.get("stats", {})
                l_range = float(l_stats.get("range", 280.0))
                l_max_charge = float(l_stats.get("charge", 0.75))
                l_cooldown = float(laser_w.get("cooldown", 0.0))
                is_shorted = t < self.shorted_until_t

                if not is_shorted and l_cooldown <= 0.05 and tdist <= l_range:
                    yaw_err = abs(angle_diff(actual_yaw, hitscan_yaw))
                    pitch_err = abs(actual_pitch - hitscan_pitch)
                    aim_aligned = (yaw_err <= hitscan_yaw_tol) and (
                        pitch_err <= hitscan_pitch_tol
                    )

                    want_swap_laser = (
                        self.desired_secondary_kind(
                            obs, self_data, zone, active_enemies, t
                        )
                        != "laser"
                        and (take_id is not None or walk_pickup is not None)
                    )

                    if l_cur_charge <= 0.0:
                        if not want_swap_laser and (has_los_to_target or tdist < 165.0):
                            fire_cmd[laser_slot] = True
                            self.laser_state = "CHARGING"
                    else:
                        req_charge = (
                            l_max_charge * 0.96
                            if combat_target["kind"] in ("crate", "npc")
                            else l_max_charge * 0.88
                        )
                        if l_cur_charge >= l_max_charge * 0.68:
                            # Priority to laser hitscan alignment as charge nears full
                            desired_yaw, desired_pitch = hitscan_yaw, hitscan_pitch

                        if l_cur_charge >= req_charge and has_los_to_target and aim_aligned:
                            # Release trigger to fire the hitscan beam at the aligned target!
                            fire_cmd[laser_slot] = False
                            self.laser_state = "RELEASING_AT_TARGET"
                        elif l_cur_charge < l_max_charge + 1.5:
                            # Hold trigger so we don't waste the beam into a wall before peeking/aligning
                            fire_cmd[laser_slot] = True
                            self.laser_state = (
                                "HOLDING_FOR_RELEASE"
                                if l_cur_charge >= req_charge
                                else "CHARGING"
                            )

            # --- LIGHTNING LOGIC ---
            if light_slot is not None and light_w is not None:
                lt_stats = light_w.get("stats", {})
                lt_range = float(lt_stats.get("range", 45.0))
                lt_cone_rad = math.radians(float(lt_stats.get("cone_deg", 30.0)))
                lt_cd = float(light_w.get("cooldown", 0.0))
                lt_cost = float(lt_stats.get("energy", 3.0))

                if (
                    lt_cd <= 0.05
                    and tdist <= lt_range - 0.5
                    and has_los_to_target
                    and energy >= lt_cost + 12.0
                ):
                    if abs(angle_diff(actual_yaw, hitscan_yaw)) <= lt_cone_rad * 0.85:
                        fire_cmd[light_slot] = True

            # --- GUN LOGIC (Coordinated with actual torso aim) ---
            if gun_slot is not None and gun_w is not None:
                g_stats = gun_w.get("stats", {})
                g_range = float(g_stats.get("range", 230.0))
                g_cd = float(gun_w.get("cooldown", 0.0))
                g_ammo = int(gun_w.get("ammo", 12))
                g_reload = float(gun_w.get("reload", 0.0))

                if (
                    g_cd <= 0.05
                    and g_reload <= 0.0
                    and g_ammo > 0
                    and tdist <= g_range
                    and has_los_to_target
                ):
                    # Fire gun whenever actual torso aim is within lead tolerance
                    if (
                        abs(angle_diff(actual_yaw, gun_yaw)) <= gun_yaw_tol
                        and abs(actual_pitch - gun_pitch) <= gun_pitch_tol
                    ):
                        fire_cmd[gun_slot] = True

            # --- MORTAR FALLBACK ---
            if (
                mortar_slot is not None
                and mortar_w is not None
                and combat_target["kind"] == "mech"
            ):
                m_stats = mortar_w.get("stats", {})
                m_speed = float(m_stats.get("speed", 72.0))
                m_grav = float(m_stats.get("gravity", 22.0))
                m_cd = float(mortar_w.get("cooldown", 0.0))
                if m_cd <= 0.05 and 32.0 <= tdist <= 210.0:
                    flight_t = tdist / max(1.0, m_speed * 0.82)
                    pred_x = tpos[0] + tvel[0] * flight_t
                    pred_y = tpos[1] + tvel[1] * flight_t
                    pred_d = math.hypot(pred_x - px, pred_y - py)
                    sin_2theta = clamp((m_grav * pred_d) / (m_speed * m_speed), -0.98, 0.98)
                    m_pitch = 0.5 * math.asin(sin_2theta)
                    m_yaw = math.atan2(pred_y - py, pred_x - px)
                    if abs(angle_diff(actual_yaw, m_yaw)) < 0.08:
                        desired_yaw, desired_pitch = m_yaw, m_pitch
                        fire_cmd[mortar_slot] = True

        else:
            # Cease-fire / Disengagement / Scavenge without offensive target:
            # 1) Face the most dangerous exposed shooter (or primary threat if hidden) to protect our 180-deg front armor arc
            facing_threat = shooters[0] if shooters else primary_enemy
            if facing_threat is not None:
                desired_yaw = math.atan2(
                    facing_threat.pos[1] - py, facing_threat.pos[0] - px
                )
                desired_pitch = 0.0
            elif (
                self.last_damaged_bearing is not None
                and (t - self.last_damaged_t) < 4.5
            ):
                desired_yaw = self.last_damaged_bearing
                desired_pitch = 0.0
            elif self.sound_pings:
                newest_snd = max(self.sound_pings, key=lambda s: s.t)
                desired_yaw = newest_snd.bearing
                desired_pitch = 0.0
            elif dist_to_goal > 2.5:
                scan_off = 0.35 * math.sin(t * 1.4)
                desired_yaw = wrap_angle(
                    math.atan2(move_goal_y - py, move_goal_x - px) + scan_off
                )
                desired_pitch = 0.0

            # 2) Handle an already-charging laser safely during cease-fire/retreat (#2):
            # Releasing `fire` FIRES the laser along `self.aim`, which can hit an enemy
            # and reset our 5s/2s regeneration delay (plus charging halves our walk speed!).
            if laser_slot is not None and laser_w is not None and l_cur_charge > 0.0:
                l_max_charge = float(laser_w.get("stats", {}).get("charge", 0.75))
                safe_yaw, safe_pitch = self.find_safe_laser_discharge_aim(
                    self_data, active_enemies, desired_yaw
                )
                # Check if our current actual torso aim is already clear of all enemies
                actual_aim_is_safe = True
                for e in active_enemies:
                    ed = math.hypot(e.pos[0] - px, e.pos[1] - py)
                    eyaw = math.atan2(e.pos[1] - py, e.pos[0] - px)
                    cone = max(0.20, math.atan2(5.5, max(4.0, ed)))
                    if abs(angle_diff(actual_yaw, eyaw)) <= cone:
                        # Also check if a wall blocks the beam before reaching enemy
                        if self.world.has_line_of_sight(
                            muzzle_pos[0],
                            muzzle_pos[1],
                            muzzle_pos[2],
                            e.pos[0],
                            e.pos[1],
                            e.pos[2] + 4.5,
                        ):
                            actual_aim_is_safe = False
                            break

                self.laser_state = "DISCHARGING_SAFE"
                if actual_aim_is_safe:
                    # Safe to release trigger now: beam misses all enemies, restores full walk speed!
                    fire_cmd[laser_slot] = False
                    desired_yaw, desired_pitch = safe_yaw, safe_pitch
                elif l_cur_charge < l_max_charge + 1.6:
                    # Keep holding trigger for 1 tick while slewing torso away from enemy
                    fire_cmd[laser_slot] = True
                    desired_yaw, desired_pitch = safe_yaw, safe_pitch
                else:
                    # About to auto-fire at hold_s limit: pitch down into ground and release
                    fire_cmd[laser_slot] = False
                    desired_yaw, desired_pitch = safe_yaw, -1.0

        # Early tactical gun reload when out of line of sight or during non-combat retreat/scavenge.
        # NEVER start a 2.4s tactical reload while we have an active combat target or a reachable
        # finish that depends on the remaining rounds in our current magazine!
        if gun_w is not None:
            g_ammo = int(gun_w.get("ammo", 12))
            g_mag = int(gun_w.get("stats", {}).get("mag", 12))
            g_reload = float(gun_w.get("reload", 0.0))
            if g_reload <= 0.0 and 0 < g_ammo < g_mag:
                if combat_target is None and not kill_imminent:
                    if self.mode in ("RETREAT", "SCAVENGE") and g_ammo <= int(
                        g_mag * 0.75
                    ):
                        want_reload = True
                    elif g_ammo <= 2 and not exposed_to_enemy:
                        want_reload = True

        # Clamp pitch to mech limits
        mech_cfg = self.config.get("mech", {})
        min_pitch = float(mech_cfg.get("min_pitch", -1.1))
        max_pitch = float(mech_cfg.get("max_pitch", 1.1))
        desired_pitch = clamp(desired_pitch, min_pitch, max_pitch)
        desired_yaw = wrap_angle(desired_yaw)

        # 8. Debug overlay
        sec_kind = self.desired_secondary_kind(obs, self_data, zone, active_enemies, t)
        mode_label = (
            f"RETREAT:{self.retreat_substate}" if self.mode == "RETREAT" else self.mode
        )
        dbg.label(
            f"{mode_label} | L:{self.laser_state} | c:{my_cubes} | w:{sec_kind}"
        )
        dbg.point([move_goal_x, move_goal_y, pz + 1.0], color="#40e0ff", size=5)
        if len(active_path) >= 2:
            prev_pt = [px, py, pz + 1.0]
            for wx, wy in active_path[:5]:
                cur_pt = [wx, wy, pz + 1.0]
                dbg.line(prev_pt, cur_pt, color="#40c0ff")
                prev_pt = cur_pt
        if primary_enemy is not None:
            dbg.line(
                [px, py, pz + 4.5],
                [
                    primary_enemy.pos[0],
                    primary_enemy.pos[1],
                    primary_enemy.pos[2] + 4.5,
                ],
                color="#ff5040",
            )

        # 9. Construct schema-valid action
        action: Dict[str, Any] = {
            "tick": obs["tick"],
            "move": move_cmd,
            "boost": want_boost,
            "jump": want_jump,
            "aim": [round(desired_yaw, 4), round(desired_pitch, 4)],
            "fire": fire_cmd,
            "reload": want_reload,
        }
        if pick_choice is not None:
            action["pick"] = int(pick_choice)
        if take_id is not None:
            action["take"] = int(take_id)
        if drop_slot is not None:
            action["drop"] = int(drop_slot)

        dbg_dict = dbg.to_dict()
        if dbg_dict:
            action["debug"] = dbg_dict

        return action


# Pin the active submission profile here or via `ENV BOT_PROFILE=...` in Dockerfile:
#   "baseline"   -> Submission 1: Reliable baseline (threat-target only, 85% HP multi-retreat)
#   "finishable" -> Submission 2: + Finishable-target selection (TTK-based firing target)
#   "duelist"    -> Submission 3: + Immediate repositioning when 2+ shooters have LOS
DEFAULT_BOT_PROFILE = "duelist"


def main() -> None:
    import os
    profile = os.environ.get("BOT_PROFILE", DEFAULT_BOT_PROFILE).strip().lower()
    if profile in ("1", "baseline"):
        bot = MobileScavengerBot(
            enable_finishable_targeting=False,
            enable_early_multi_reposition=False,
        )
    elif profile in ("2", "finishable"):
        bot = MobileScavengerBot(
            enable_finishable_targeting=True,
            enable_early_multi_reposition=False,
        )
    else:
        bot = MobileScavengerBot(
            enable_finishable_targeting=True,
            enable_early_multi_reposition=True,
        )
    run_bot(
        bot_name="nanite-duelist",
        on_hello=bot.on_hello,
        on_obs=bot.on_obs,
        on_end=bot.on_end,
    )


if __name__ == "__main__":
    main()
