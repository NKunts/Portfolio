# =============================================================================
# Mission 5: SI Units & Schwarzschild Radius (Skeleton)
# =============================================================================
# This skeleton provides the structure for using SI units and the Schwarzschild radius in the simulation.
# Implement the logic for unit conversion, black hole radius calculation, and beam management.
# The vertex shader setup from previous missions is available for reuse.
# -----------------------------------------------------------------------------
# What this mission adds on top of Mission 4:
#   • A world described in SI units (meters, seconds, kilograms).
#   • Compute the Schwarzschild radius r_s = 2GM/c^2 from a chosen mass (kg).
#   • Convert meters <-> pixels using a scale meters_per_pixel (m/px).
#   • Make the on-screen black-hole radius self.rs_px come from r_s (in meters).
#   • Make beam speed derived from the speed of light c (scaled down to be visible).
#   • Optionally lock the grid spacing to a “nice” metric step so grid lines
#     correspond to fixed distances in meters on screen.
#
# Reuse from previous missions:
#   • Background rendering (Mission 1)
#   • Single-point shader and POINTS rendering (Mission 2)
#   • Multiple beams with one VBO (Mission 3)
#   • Per-vertex color, collision flash, and statistics (Mission 4)
#
# Controls added in this mission (we also wire these into MissionControl polling):
#   Z / X  : zoom in / out   (change meters_per_pixel -> BH appears bigger/smaller)
#   M / N  : mass up / down  (changes r_s in meters -> updates BH radius in pixels)
#   F / G  : slower / faster world time (scales beam speed derived from c)
#   V      : toggle “grid locked to meters” (auto-picks a nice metric step)
#   P      : print current SI parameters to the console (debug)
#
# Notes on units:
#   • meters_per_pixel (mpp) tells how many meters map to one screen pixel.
#   • Schwarzschild radius in meters: r_s_m = 2 * G * mass_kg / c^2
#   • On screen radius in pixels:     rs_px = r_s_m / meters_per_pixel
#   • Beam speed (px/s):              beam_speed_px = (c / mpp) * world_speed_factor
#     (world_speed_factor << 1 so motion is visible on a small screen)
# =============================================================================

import numpy as np
import moderngl
from .mission4_multiple_beams import Mission4MultipleBeams


class Mission5UnitsSchwarzschild(Mission4MultipleBeams):
    """
    Multiple parallel light beams using SI units with a physically computed BH radius.
    """

    # Physical constants (SI units)
    G = 6.67430e-11       # m^3 kg^-1 s^-2
    c = 299_792_458.0     # m/s
    M_SUN = 1.988_47e30   # kg (solar mass)

    def get_name(self) -> str:
        return "Mission 5: SI Units & Schwarzschild Radius"

    # -------------------------------------------------------------------------
    # 1) Initialization of SI world
    # -------------------------------------------------------------------------
    def initialize(self) -> None:
        """
        Initialize the scene as in Mission 4, then introduce SI-world parameters.
        The base class (M4) sets:
          • multiple beams (positions VBO), colors VBO, collision flash
          • point-sprite shader with per-vertex color (pt4_prog)
          • grid, BH center (pixels)
        Here we add:
          • self.mass_kg, self.meters_per_pixel, self.world_speed_factor
          • compute Schwarzschild radius in meters and convert to pixels
          • optional “grid locked to meters” mode that auto-chooses grid spacing
        """
        super().initialize()  # builds everything from Mission 4

        # --- SI world state (pick readable defaults) ----------------------
        # Choose a stellar-mass black hole (e.g., 50 M_sun). This yields r_s ≈ 147.5 km.
        self.mass_kg: float = 50.0 * self.M_SUN

        # meters_per_pixel (m/px): how many meters correspond to 1 screen pixel.
        # With ~1000 m/px, a 147.5 km Schwarzschild radius gives ~147 px on screen.
        self.meters_per_pixel: float = 1000.0

        # Time scaling: we don't want beams to move at 300,000 km/s on a tiny window.
        # We scale c down by a factor so the motion is visible but still “proportional”.
        self.world_speed_factor: float = 0.006  # try 0.003 .. 0.02 for your taste

        # Lock grid to nice metric steps by default (toggle with V)
        self.grid_locked_to_meters: bool = True

        # Compute derived quantities and push into pixel-based state (self.rs_px, beam speed)
        self._recompute_from_units()

        # Optional: print initial setup so the learner sees concrete numbers
        self._print_units("Initialized")

    # -------------------------------------------------------------------------
    # 2) Per-frame update
    # -------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        """
        Keep the pixel-based simulation from M4, but **update pixel values** derived
        from SI state each frame, so any key change (mass, zoom, time scaling)
        immediately affects:
          • self.rs_px         (used for BH disc/collision)
          • self.beam_speed_px (used for beam motion)
          • self.grid_gap_px   (if grid is locked to meters)
        Then call the parent update to move beams and do collisions/flash.
        """
        # Update the pixel world derived from SI world
        self._recompute_from_units()

        # Now run Mission 4 logic (move beams, check collisions, update flash/colors)
        super().update(dt)

    # -------------------------------------------------------------------------
    # 3) Rendering (nothing special needed; parent already draws everything)
    # -------------------------------------------------------------------------
    def render(self) -> None:
        """
        We still render background with BH disc + beams with per-vertex color exactly
        as in Mission 4. The only difference is that self.rs_px / grid spacing / beam
        speed now come from SI conversions performed in update().
        """
        super().render()

    # -------------------------------------------------------------------------

    # Helpers: recompute SI-derived values and pretty-print them
    # -------------------------------------------------------------------------
    def _recompute_from_units(self) -> None:
        """Пересчёт всех производных величин из текущих SI-настроек."""
        # Гарантируем наличие констант и базовых полей
        self.G = getattr(self, "G", 6.67430e-11)
        self.c = getattr(self, "c", 299_792_458.0)
        self.mass_kg = float(getattr(self, "mass_kg", 50.0 * self.M_SUN))
        self.meters_per_pixel = float(getattr(self, "meters_per_pixel", 1000.0))
        self.world_speed_factor = float(getattr(self, "world_speed_factor", 0.006))
        self.grid_locked_to_meters = bool(getattr(self, "grid_locked_to_meters", True))

        # Производные величины
        self.rs_m  = 2.0 * self.G * self.mass_kg / (self.c * self.c)
        self.rs_px = self.rs_m / self.meters_per_pixel
        self.c_px_s        = self.c / self.meters_per_pixel
        self.beam_speed_px = self.c_px_s * self.world_speed_factor

        # time_scale используют M6/M7 (оставим, если уже есть)
        self.time_scale = float(getattr(self, "time_scale", 1.0))

        # Шаг сетки
        if self.grid_locked_to_meters:
            grid_gap_m = self._choose_nice_metric_grid(self.meters_per_pixel)
            self.grid_gap_px = grid_gap_m / self.meters_per_pixel
        else:
            self.grid_gap_px = float(getattr(self, "grid_gap_px", 32.0))

    def _print_units(self, why: str = "Initialized") -> None:
        """Отладочная печать текущих SI-параметров (для консоли)."""
        try:
            mass_msun = self.mass_kg / self.M_SUN
            print(
                f"[M5] {why}  mass = {self.mass_kg:.3e} kg  ({mass_msun:.2f} M_sun) | "
                f"m/px = {self.meters_per_pixel:.3e} | "
                f"r_s = {self.rs_m:.3e} m ({self.rs_px:.1f} px) | "
                f"c* = {self.c_px_s:.1f} px/s"
            )
        except Exception:
            print(f"[M5] {why} (units state incomplete)")

    def _choose_nice_metric_grid(self, mpp: float) -> float:
        """Подбирает шаг сетки 1/2/5×10^k м так, чтобы было ≈64 px между линиями."""
        target_px = 64.0
        target_m = max(1e-9, target_px * mpp)
        base = 10.0 ** np.floor(np.log10(target_m))
        candidates = np.array([1.0, 2.0, 5.0, 10.0]) * base
        return float(candidates[np.argmin(np.abs(candidates - target_m))])

    # -------------------------------------------------------------------------
    # Methods the control panel calls (no key handling inside the mission!)
    # -------------------------------------------------------------------------
    def zoom_in(self) -> None:
        self.meters_per_pixel = max(1.0, self.meters_per_pixel / 1.2)

    def zoom_out(self) -> None:
        self.meters_per_pixel = self.meters_per_pixel * 1.2

    def mass_up(self) -> None:
        self.mass_kg = self.mass_kg * 1.1

    def mass_down(self) -> None:
        self.mass_kg = max(1e20, self.mass_kg / 1.1)

    def slower(self) -> None:
        self.world_speed_factor = max(1e-5, self.world_speed_factor / 1.2)

    def faster(self) -> None:
        self.world_speed_factor = min(1.0, self.world_speed_factor * 1.2)

    def toggle_grid_lock(self) -> None:
        self.grid_locked_to_meters = not self.grid_locked_to_meters


    def _print_units(self, why: str = "Initialized") -> None:
        """Отладочная печать текущих SI-параметров (для консоли)."""
        try:
            mass_msun = self.mass_kg / self.M_SUN
            print(
                f"[M5] {why}  mass = {self.mass_kg:.3e} kg  ({mass_msun:.2f} M_sun) | "
                f"m/px = {self.meters_per_pixel:.3e} | "
                f"r_s = {self.rs_m:.3e} m ({self.rs_px:.1f} px) | "
                f"c* = {self.c_px_s:.1f} px/s"
            )
        except Exception:
            print(f"[M5] {why} (units state incomplete)")

    def _choose_nice_metric_grid(self, mpp: float) -> float:
        """
        Возвращает «красивый» метрический шаг 1/2/5×10^k метров при заданном m/px,
        чтобы расстояния между линиями сетки были ~64 px.
        """
        target_px = 64.0
        target_m = max(1e-9, target_px * mpp)
        base = 10.0 ** np.floor(np.log10(target_m))
        candidates = np.array([1.0, 2.0, 5.0, 10.0]) * base
        return float(candidates[np.argmin(np.abs(candidates - target_m))])

    # -------------------------------------------------------------------------
    # Methods the control panel calls (no key handling inside the mission!)
    # -------------------------------------------------------------------------
    def zoom_in(self) -> None:
        """Увеличить (уменьшить m/px)."""
        self.meters_per_pixel = max(1.0, self.meters_per_pixel / 1.2)

    def zoom_out(self) -> None:
        """Отдалить (увеличить m/px)."""
        self.meters_per_pixel = self.meters_per_pixel * 1.2

    def mass_up(self) -> None:
        """Увеличить массу на ~10%."""
        self.mass_kg = self.mass_kg * 1.1

    def mass_down(self) -> None:
        """Уменьшить массу на ~10% (с ограничением снизу)."""
        self.mass_kg = max(1e20, self.mass_kg / 1.1)

    def slower(self) -> None:
        """Замедлить время (уменьшить world_speed_factor)."""
        self.world_speed_factor = max(1e-5, self.world_speed_factor / 1.2)

    def faster(self) -> None:
        """Ускорить время (увеличить world_speed_factor)."""
        self.world_speed_factor = min(1.0, self.world_speed_factor * 1.2)

    def toggle_grid_lock(self) -> None:
        """Переключить режим «сетка привязана к метрам»."""
        self.grid_locked_to_meters = not self.grid_locked_to_meters

    # -------------------------------------------------------------------------
    # 4) Key handling for SI parameters (works with event + with controller polling)
    # -------------------------------------------------------------------------
    def handle_key(self, key, action, modifiers, keys) -> None:
        pass


