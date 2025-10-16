# =============================================================================
# Mission 9: Gravitational Redshift Visualization (overlay, depth-safe)
# =============================================================================
# Draws colored photon heads on top of the base Mission7/8 renderer.
# Guarantees visibility by disabling depth test and using our own shader/VAO.
# Adds single-beam-friendly auto-contrast using a running g-range with decay.
# =============================================================================

import numpy as np
import moderngl
from .mission8_validation import Mission8Validation


class Mission9Redshift(Mission8Validation):
    def get_name(self):
        return "Mission 9: Gravitational Redshift Visualization"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def initialize(self):
        # Reuse physics, trails, etc. from M7/M8.
        super().initialize()

        # Visualization params
        self.redshift_enabled = True
        self.auto_contrast = True          # normalize colors dynamically
        self.g_min = 0.55                  # used in absolute mode only
        self.g_clip = (0.001, 1.0)         # numeric clamp for g
        self._overlay_point_size = float(getattr(self, "trail_point_size_px", 6.0)) * 1.35

        # Running g-range for single/low beam counts (decaying window)
        self._g_lo = 1.0                   # running min of g
        self._g_hi = 0.0                   # running max of g
        self._g_decay = 0.985              # decay per update towards current frame

        # CPU buffers
        n = int(self.beam_count)
        self._m9_pos = np.zeros((n, 2), dtype=np.float32)
        self._m9_rgba = np.ones((n, 4), dtype=np.float32)

        # Minimal overlay shader (Pixel -> NDC). Depth is irrelevant; we disable test.
        self._m9_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 a_pos;
                in vec4 a_color;
                out vec4 v_color;
                uniform vec2 u_view;   // window size in pixels
                uniform float u_ptsz;  // point size in pixels
                void main() {
                    vec2 ndc = vec2(
                        (a_pos.x / u_view.x) * 2.0 - 1.0,
                        (a_pos.y / u_view.y) * 2.0 - 1.0
                    );
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    gl_PointSize = u_ptsz;
                    v_color = a_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 fragColor;
                void main() {
                    // Soft round point
                    vec2 p = gl_PointCoord * 2.0 - 1.0;
                    float d = dot(p, p);
                    if (d > 1.0) discard;
                    float alpha = 1.0 - smoothstep(0.85, 1.0, sqrt(d));
                    fragColor = vec4(v_color.rgb, v_color.a * alpha);
                }
            """,
        )

        # GPU buffers + VAO
        self._m9_pos_bo = self.ctx.buffer(self._m9_pos.tobytes())
        self._m9_col_bo = self.ctx.buffer(self._m9_rgba.tobytes())
        self._m9_vao = self.ctx.vertex_array(
            self._m9_prog,
            [
                (self._m9_pos_bo, "2f", "a_pos"),
                (self._m9_col_bo, "4f", "a_color"),
            ],
        )

        # Optional: longer trails for nicer visuals
        try:
            self.trail_enabled = True
            if hasattr(self, "increase_trail_len"):
                for _ in range(2):
                    self.increase_trail_len()
        except Exception:
            pass

        print("[M9] initialize() — ready (redshift overlay + robust auto-contrast)")

    def update(self, dt):
        # Advance physics (M7/M8)
        super().update(dt)
        if not (hasattr(self, "_positions") and isinstance(self._positions, np.ndarray)):
            return
        if not self.redshift_enabled:
            return

        # Copy head positions (pixels)
        self._m9_pos[:, :] = self._positions[:, :2].astype(np.float32)

        # Radius from BH center (pixels)
        cx, cy = float(self.center[0]), float(self.center[1])
        dx = self._m9_pos[:, 0].astype(np.float64) - cx
        dy = self._m9_pos[:, 1].astype(np.float64) - cy
        r = np.hypot(dx, dy)

        # Schwarzschild radius in pixels
        rs = float(getattr(self, "rs_px", 80.0))

        # g(r) = sqrt(1 - rs / r), clamped
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.sqrt(np.clip(1.0 - (rs / np.maximum(r, rs + 1e-6)), self.g_clip[0], 1.0))

        # Color mapping
        if self.auto_contrast:
            # Per-frame spread
            gmin_f = float(np.min(g))
            gmax_f = float(np.max(g))

            # Update running range with gentle decay (helps single-beam cases)
            self._g_lo = min(gmin_f, self._g_lo * self._g_decay + gmin_f * (1.0 - self._g_decay))
            self._g_hi = max(gmax_f, self._g_hi * self._g_decay + gmax_f * (1.0 - self._g_decay))

            # Ensure a usable span
            span = max(1e-4, self._g_hi - self._g_lo)

            # Normalize to [0,1]
            t = (g - self._g_lo) / span
            t = np.clip(t, 0.0, 1.0)
            rgb = self._map_t_to_rgb(t)
        else:
            # Absolute mapping against fixed g_min
            rgb = self._map_g_to_rgb(g, g_min=self.g_min)

        # Fill RGBA and upload to GPU
        self._m9_rgba[:, :3] = rgb.astype(np.float32)
        self._m9_rgba[:, 3] = 1.0
        self._m9_pos_bo.write(self._m9_pos.tobytes())
        self._m9_col_bo.write(self._m9_rgba.tobytes())

    def render(self):
        # Draw base scene first
        super().render()

        # Draw colored heads on top (no depth test)
        try:
            self.ctx.disable(moderngl.DEPTH_TEST)
        except Exception:
            pass
        try:
            self.ctx.enable(moderngl.BLEND)
        except Exception:
            pass

        self._m9_prog["u_view"].value = (float(self.width), float(self.height))
        self._m9_prog["u_ptsz"].value = float(self._overlay_point_size)
        self._m9_vao.render(mode=moderngl.POINTS, vertices=int(self.beam_count))

    def handle_key(self, key, action, modifiers, keys):
        # Keep base hotkeys
        super().handle_key(key, action, modifiers, keys)
        if action != getattr(keys, "ACTION_PRESS", action):
            return

        # Helper: robust "is this key X?" for different backends
        def _is(k, names):
            if getattr(k, "name", None) in names:
                return True
            for nm in names:
                if getattr(keys, nm, None) == k:
                    return True
            return False

        # Reseed rays
        if _is(key, ("R", "KEY_R")) and hasattr(self, "_reseed_geodesics"):
            print("[M9] Reseed geodesics")
            self._reseed_geodesics(keep_trails=False)

        # Toggle auto-contrast (works even without control-panel binding)
        if _is(key, ("G", "KEY_G")):
            self.auto_contrast = not self.auto_contrast
            mode = "auto-contrast" if self.auto_contrast else "absolute"
            print(f"[M9] Color mode: {mode}")

        # Adjust g_min only in absolute mode
        if not self.auto_contrast:
            if _is(key, ("LEFT_BRACKET",)):
                self.g_min = min(0.95, max(0.05, float(self.g_min) + 0.05))
                print(f"[M9] g_min -> {self.g_min:.2f}")
            if _is(key, ("RIGHT_BRACKET",)):
                self.g_min = min(0.95, max(0.05, float(self.g_min) - 0.05))
                print(f"[M9] g_min -> {self.g_min:.2f}")

    # ------------------------------------------------------------------ #
    # Color mapping helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _map_g_to_rgb(g: np.ndarray, g_min: float = 0.5) -> np.ndarray:
        """Absolute mapping: g ∈ [g_min,1] -> gradient red→yellow→blue-white."""
        g = np.asarray(g, dtype=np.float64)
        t = (g - float(g_min)) / max(1e-6, (1.0 - float(g_min)))
        t = np.clip(t, 0.0, 1.0)
        return Mission9Redshift._map_t_to_rgb(t)

    @staticmethod
    def _map_t_to_rgb(t: np.ndarray) -> np.ndarray:
        """t ∈ [0,1] -> piecewise-linear gradient (red→yellow→blue-white)."""
        t = np.asarray(t, dtype=np.float64)
        c0 = np.array([1.00, 0.10, 0.10], dtype=np.float64)  # deep red
        c1 = np.array([1.00, 0.80, 0.20], dtype=np.float64)  # orange/yellow
        c2 = np.array([0.85, 0.90, 1.00], dtype=np.float64)  # soft blue-white
        mid = 0.5

        out = np.empty((t.shape[0], 3), dtype=np.float64)
        lo = t <= mid
        hi = ~lo
        if lo.any():
            tl = t[lo] / mid
            out[lo] = (1.0 - tl)[:, None] * c0 + tl[:, None] * c1
        if hi.any():
            th = (t[hi] - mid) / (1.0 - mid)
            out[hi] = (1.0 - th)[:, None] * c1 + th[:, None] * c2
        return out.astype(np.float32)
