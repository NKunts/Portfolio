# =============================================================================
# Mission 7: Light Bending (Schwarzschild Geodesics) Skeleton
# =============================================================================
# This skeleton provides the structure for integrating Schwarzschild null geodesics in the equatorial plane.
# Implement the logic for geodesic integration, ray management, and rendering.
# The vertex shader setup from previous missions is available for reuse.
# =============================================================================
# We integrate null geodesics in the Schwarzschild metric in the equatorial plane.
# Equation (meters):  d²u/dφ² + u = 3 M u²,  with u = 1/r
# M is the geometric mass: M = G * M_kg / c²  [meters]
# We step in φ using RK4, convert (r, φ) to pixels, update heads and trails.
# Rendering (grid, BH disc, heads+trails) is reused from Mission 6.
# =============================================================================

import numpy as np
from .mission6_fixed_timestep import Mission6FixedTimestep

class Mission7LightBending(Mission6FixedTimestep):
    """Light bending in Schwarzschild spacetime (equatorial null geodesics)."""

    # Good visual defaults
    DEFAULT_BEAM_COUNT = 31
    DEFAULT_BEAM_SPACING_PX = 10.0
    DEFAULT_PHI0_DEG = -12.0      # start far to the right
    DEFAULT_PHIMAX_DEG = 168.0    # sweep well past the BH

    def get_name(self):
        return "Mission 7: Light Bending (Schwarzschild Null Geodesics)"
    
    def _recompute_from_units(self, reason: str = "M7 shim"):
        """
        Minimal units recomputation shim so that Mission5.initialize() doesn't fail.
        We ensure mass, Schwarzschild radius, meters-per-pixel and c* in px/s exist.
        """
        # Physical constants (keep same names parent expects)
        G = getattr(self, "G", 6.67430e-11)
        c = getattr(self, "c", 299_792_458)

        # Mass fallback: 50 solar masses (matches earlier logs)
        if not hasattr(self, "mass_kg"):
            self.mass_kg = 50.0 * 1.98847e30  # kg

        # Schwarzschild radius (meters) and geometric mass (meters)
        self.rs_m  = 2.0 * G * self.mass_kg / (c * c)
        self.M_geo = G * self.mass_kg / (c * c)

        # Pixel scale: prefer existing meters_per_pixel; otherwise derive from rs_px if present,
        # else use a sane default so the scene is visible.
        if hasattr(self, "meters_per_pixel"):
            m_per_px = float(self.meters_per_pixel)
        elif hasattr(self, "rs_px"):
            rs_px = max(4.0, float(self.rs_px))
            m_per_px = max(self.rs_m / rs_px, 1.0)
        else:
            m_per_px = 1_000.0  # default visual scale

        self.meters_per_pixel = float(m_per_px)

        # Convenience: speed of light in pixels/sec for other missions' UI text
        self.c_star_px = float(c / self.meters_per_pixel)

        # Optional diagnostic (kept short so it doesn't spam)
        print(f"[M7] Units shim — mass={self.mass_kg:.3e} kg | m/px={self.meters_per_pixel:.3e} | r_s={self.rs_m:.3e} m")


    # ---------------------------- RK4 integrator -----------------------------
    @staticmethod
    def _rk4_step(u, up, M_geo, dphi):
        """
        One RK4 step for:
            u'  = up
            up' = -u + 3 M u^2
        Returns (u_next, up_next).
        """
        def acc(cur_u):
            return -cur_u + 3.0 * M_geo * (cur_u * cur_u)

        k1_u  = up
        k1_up = acc(u)

        k2_u  = up + 0.5 * dphi * k1_up
        k2_up = acc(u + 0.5 * dphi * k1_u)

        k3_u  = up + 0.5 * dphi * k2_up
        k3_up = acc(u + 0.5 * dphi * k2_u)

        k4_u  = up + dphi * k3_up
        k4_up = acc(u + dphi * k3_u)

        u_next  = u  + (dphi / 6.0) * (k1_u  + 2.0 * k2_u  + 2.0 * k3_u  + k4_u)
        up_next = up + (dphi / 6.0) * (k1_up + 2.0 * k2_up + 2.0 * k3_up + k4_up)
        return u_next, up_next

    def _preset_angles(self):
        if not hasattr(self, "phi0"):
            self.phi0 = np.deg2rad(getattr(self, "DEFAULT_PHI0_DEG", -12.0))
        if not hasattr(self, "phi_max"):
            self.phi_max = np.deg2rad(getattr(self, "DEFAULT_PHIMAX_DEG", 168.0))

    def _safe_init_layout(self):
        """Ensure beam layout arrays exist with reasonable defaults."""
        if not hasattr(self, "beam_count"):
            self.beam_count = int(self.DEFAULT_BEAM_COUNT)
        if not hasattr(self, "beam_spacing_px"):
            self.beam_spacing_px = float(self.DEFAULT_BEAM_SPACING_PX)
        if not hasattr(self, "_positions") or self._positions.shape != (self.beam_count, 2):
            self._positions = np.zeros((self.beam_count, 2), dtype="f4")

    def _reseed_geodesics(self, keep_trails: bool = False):
        """
        Rebuild per-ray state with current phi window and stored impact params _b_m.
        If keep_trails is False, snap trail ring to the new heads for instant visibility.
        """
        u0  = np.sin(self.phi0) / self._b_m
        up0 = np.cos(self.phi0) / self._b_m

        self._phi      = np.full(self.beam_count, self.phi0, dtype=np.float32)
        self._u        = u0.astype(np.float32)
        self._up       = up0.astype(np.float32)
        self._alive    = np.ones(self.beam_count, dtype=bool)
        self._finished = np.zeros(self.beam_count, dtype=bool)

        # Place heads
        r0_m  = 1.0 / np.maximum(self._u, 1e-12)
        r0_px = r0_m / float(self.meters_per_pixel)
        self._positions[:, 0] = self.center[0] + r0_px * np.cos(self._phi)
        self._positions[:, 1] = self.center[1] + r0_px * np.sin(self._phi)
        if hasattr(self, "_vbo"):
            self._vbo.write(self._positions.tobytes())

        # Trails
        if not keep_trails and hasattr(self, "trail_len") and hasattr(self, "trail_positions"):
            self.trail_head = 0
            self.trail_positions[...] = self._positions[None, :, :].astype("f4")
            if hasattr(self, "_trail_pos_vbo") and hasattr(self, "_trail_age_vbo"):
                self._push_trails_to_gpu_full_snapshot()

    def set_loop(self, on: bool):
        """Toggle looping (wrap at phi_max or horizon)."""
        self.loop_rays = bool(on)

    # ------------------------------- Initialize ------------------------------
    def initialize(self):
        """
        Build everything from Mission 6 (background, beams, trail pipeline, SI conversions),
        then prepare geodesic state: φ, u=1/r, du/dφ per ray; impact parameters from spacing.
        """
        super().initialize()  # grid/BH, VAOs/VBOs, trails, fixed timestep defaults

        # SI fields
        G = getattr(self, "G", 6.67430e-11)
        c = getattr(self, "c", 299_792_458)
        if not hasattr(self, "meters_per_pixel"):
            self.meters_per_pixel = 1_000.0
        if not hasattr(self, "mass_kg"):
            self.mass_kg = 5.0e30

        # Schwarzschild scale
        self.rs_m  = 2.0 * G * self.mass_kg / (c * c)     # meters
        self.M_geo = G * self.mass_kg / (c * c)           # meters
        self.b_crit_m = 3.0 * np.sqrt(3.0) * self.M_geo   # capture threshold for photons

        # Layout + angles
        self._safe_init_layout()
        self._preset_angles()
        base_rate     = 1.5
        self.phi_rate = float(getattr(self, "time_scale", 1.0)) * base_rate  # rad/s
        self.loop_rays = True

        # Signed impact parameter from vertical spacing
        idx = np.arange(self.beam_count, dtype=np.float32)
        y_offsets_px = (idx - 0.5 * (self.beam_count - 1)) * float(self.beam_spacing_px)
        b_signed = y_offsets_px * float(self.meters_per_pixel)

        b_min = 1.10 * self.b_crit_m
        sign = np.sign(b_signed)
        sign[sign == 0.0] = 1.0
        b_m = np.where(np.abs(b_signed) < b_min, sign * b_min, b_signed)
        self._b_m = b_m.astype(np.float32)

        # Seed state and trails
        self._reseed_geodesics(keep_trails=False)

        # Fixed-step safety
        if not hasattr(self, "fixed_dt_s"):   self.fixed_dt_s = 1.0 / 240.0
        if not hasattr(self, "_accum_s"):     self._accum_s   = 0.0
        if not hasattr(self, "max_substeps"): self.max_substeps = 8

    # ---------------------------- Trail helpers ------------------------------
    def _trail_write_single(self, beam_index: int):
        """
        Append current head position of beam 'beam_index' into the ring row 'trail_head'.
        When the last beam is written, upload pos+age arrays and advance the circular head.
        """
        if not hasattr(self, "trail_positions") or not hasattr(self, "trail_len"):
            return

        self.trail_positions[self.trail_head, beam_index, :] = self._positions[beam_index, :]

        if beam_index == (self.beam_count - 1):
            if hasattr(self, "_trail_pos_vbo") and hasattr(self, "_trail_age_vbo"):
                pos = self.trail_positions.reshape(self.trail_len * self.beam_count, 2)
                rows = (np.arange(self.trail_len) - self.trail_head) % self.trail_len
                age_per_row = rows.astype(np.float32) / float(self.trail_len - 1 if self.trail_len > 1 else 1)
                age = np.repeat(age_per_row[:, None], self.beam_count, axis=1).reshape(-1).astype("f4")
                self._trail_pos_vbo.write(pos.astype("f4").tobytes())
                self._trail_age_vbo.write(age.tobytes())
            self.trail_head = (self.trail_head + 1) % int(self.trail_len)

    def _push_trails_to_gpu_full_snapshot(self):
        """Upload the entire trail ring buffer to the GPU once (safe-guarded)."""
        if not (hasattr(self, "trail_positions") and hasattr(self, "trail_len")
                and hasattr(self, "beam_count")
                and hasattr(self, "_trail_pos_vbo") and hasattr(self, "_trail_age_vbo")):
            return
        pos = self.trail_positions.reshape(self.trail_len * self.beam_count, 2)
        rows = (np.arange(self.trail_len) - self.trail_head) % self.trail_len
        age_per_row = rows.astype(np.float32) / float(self.trail_len - 1 if self.trail_len > 1 else 1)
        age = np.repeat(age_per_row[:, None], self.beam_count, axis=1).reshape(-1).astype("f4")
        self._trail_pos_vbo.write(pos.astype("f4").tobytes())
        self._trail_age_vbo.write(age.tobytes())

    # ------------------------ Fixed-timestep update --------------------------
    def _physics_substep(self):
        """
        One fixed-Δt physics tick:
        - Δφ = φ_rate * fixed_dt_s
        - RK4 step for (u, du/dφ)
        - capture if r <= (1+ε) r_s (optionally loop)
        - write head position and update the trail ring buffer
        """
        dphi = float(self.phi_rate) * float(self.fixed_dt_s)
        M = float(self.M_geo)
        eps_capture = 1.001

        for i in range(self.beam_count):
            if not (self._alive[i] and not self._finished[i]):
                continue

            # Integrate in φ
            u_i, up_i = float(self._u[i]), float(self._up[i])
            u_i, up_i = self._rk4_step(u_i, up_i, M, dphi)
            phi_i = float(self._phi[i] + dphi)

            # Radius from u (guard)
            r_m = (1.0 / u_i) if u_i > 0.0 else 1e12

            # Horizon capture
            if r_m <= eps_capture * float(self.rs_m):
                if getattr(self, "loop_rays", False):
                    # Restart at φ0 with same signed b
                    self._phi[i] = float(self.phi0)
                    b = float(self._b_m[i])
                    self._u[i]  = np.sin(self.phi0) / b
                    self._up[i] = np.cos(self.phi0) / b
                    r0_px = (1.0 / max(self._u[i], 1e-12)) / float(self.meters_per_pixel)
                    self._positions[i, 0] = float(self.center[0]) + r0_px * np.cos(self._phi[i])
                    self._positions[i, 1] = float(self.center[1]) + r0_px * np.sin(self._phi[i])
                    self._trail_write_single(i)
                    continue
                else:
                    # Freeze just above horizon and stop
                    self._alive[i] = False
                    r_px = (eps_capture * float(self.rs_m)) / float(self.meters_per_pixel)
                    self._positions[i, 0] = float(self.center[0]) + r_px * np.cos(phi_i)
                    self._positions[i, 1] = float(self.center[1]) + r_px * np.sin(phi_i)
                    self._trail_write_single(i)
                    continue

            # Exit visual window
            if phi_i >= self.phi_max:
                if getattr(self, "loop_rays", False):
                    self._phi[i] = float(self.phi0)
                    b = float(self._b_m[i])
                    self._u[i]  = np.sin(self.phi0) / b
                    self._up[i] = np.cos(self.phi0) / b
                    r0_px = (1.0 / max(self._u[i], 1e-12)) / float(self.meters_per_pixel)
                    self._positions[i, 0] = float(self.center[0]) + r0_px * np.cos(self._phi[i])
                    self._positions[i, 1] = float(self.center[1]) + r0_px * np.sin(self._phi[i])
                    self._trail_write_single(i)
                    continue
                else:
                    self._finished[i] = True
                    continue

            # Normal head update
            r_px = r_m / float(self.meters_per_pixel)
            self._positions[i, 0] = float(self.center[0]) + r_px * np.cos(phi_i)
            self._positions[i, 1] = float(self.center[1]) + r_px * np.sin(phi_i)

            # Commit state
            self._phi[i] = phi_i
            self._u[i]   = u_i
            self._up[i]  = up_i

            # Trail
            self._trail_write_single(i)

        if hasattr(self, "_vbo"):
            self._vbo.write(self._positions.tobytes())

    def _m7_reset():
        print("[M7] Reseed geodesics with current φ-window (no re-init)")
        if hasattr(m, "_reseed_geodesics"):
            m._reseed_geodesics(keep_trails=False) 
    
    def update(self, dt):
        """Fixed-timestep accumulator loop."""
        if not hasattr(self, "fixed_dt_s"):
            self.fixed_dt_s = 1.0 / 240.0
        if not hasattr(self, "_accum_s"):
            self._accum_s = 0.0
        if not hasattr(self, "max_substeps"):
            self.max_substeps = 8

        self._accum_s += float(dt)
        steps = 0
        while self._accum_s >= self.fixed_dt_s and steps < self.max_substeps:
            self._accum_s -= self.fixed_dt_s
            steps += 1
            self._physics_substep()

    # ------------------------------ Rendering --------------------------------
    def render(self):
        """Reuse Mission 6 pipeline (trails, heads, BH disc, grid)."""
        super().render()
