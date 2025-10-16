# =============================================================================
# Mission 8: Light Bending Validation
# =============================================================================
# Collects far-field samples of geodesics, fits inbound/outbound asymptotes,
# computes numeric deflection and compares it to the weak-field analytic
# prediction δ ≈ 4*M_geo/|b|.
# Rendering is reused from Missions 6/7 (grid, BH disc, heads, trails).
# =============================================================================

import time
import numpy as np
from .mission7_light_bending import Mission7LightBending


class Mission8Validation(Mission7LightBending):
    """
    Numeric validation of light bending against the analytic weak-field formula.
    """

    # Tuning constants
    FIT_SAMPLES       = 64     # rolling samples kept per asymptote window
    MIN_FIT_SAMPLES   = 12     # minimum points to run a fit
    PHI_WINDOW_RAD    = 0.18   # φ-window near start/end to collect samples
    SUMMARY_PERIOD_S  = 1.0    # throttle console summary
    FAR_RADIUS_FACTOR = 0.80   # far radius as fraction of min(view_w, view_h)
    PHI0_DEG          = -12.0  # enforced φ0 for validation (deg)
    PHI_MAX_DEG       = 179.5  # enforced φ_max for validation (deg)

    def get_name(self):
        return "Mission 8: Light Bending Validation"

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def initialize(self):
        """Build M7 pipeline, then set up validation state."""
        print("[M8] initialize() — entering")

        # 1) Full Mission 7 setup (grid/BH, fixed timestep, trails, geodesics)
        super().initialize()
        print("[M8] initialize() — super().initialize() done")

        # 2) Visuals
        self.trail_enabled = True
        if hasattr(self, "show_grid"):
            self.show_grid = True

        # 3) One pass only for validation: no respawn/loop
        self.set_loop(False)

        # 4) Enforce a wide φ window for robust asymptotes
        self.phi0    = np.deg2rad(self.PHI0_DEG)
        self.phi_max = np.deg2rad(self.PHI_MAX_DEG)

        # 5) Reseed geodesics so heads/trails reflect the current window and loop state
        if hasattr(self, "_reseed_geodesics"):
            self._reseed_geodesics(keep_trails=False)

        # 6) Per-beam measurement buffers and far radius
        cnt = int(self.beam_count)
        self._meas = [{
            "in_pts":  np.empty((0, 2), dtype=np.float32),
            "out_pts": np.empty((0, 2), dtype=np.float32),
            "done":    False,
            "result":  None,
        } for _ in range(cnt)]

        self._far_radius_px = float(self.FAR_RADIUS_FACTOR) * float(min(self.width, self.height))
        self._last_summary_t = 0.0

        print("[M8] initialize() — ready (loop=False, trails ON, grid ON)")
        print("[M8] Validation: collecting asymptotes and comparing to 4*M_geo/|b|")

    def update(self, dt):
        """Run physics; collect far-field samples; compute deflection when ready."""
        super().update(dt)

        # Guard: if geodesic state is not ready yet
        if not (hasattr(self, "_phi") and hasattr(self, "_positions")):
            if not hasattr(self, "_m8_warned_no_state"):
                print("[M8] update() — no geodesic state yet")
                self._m8_warned_no_state = True
            return

        self._collect_far_field_samples()
        self._compute_deflections_if_ready()

        # Lightweight heartbeat
        now = time.time()
        if now - getattr(self, "_last_ping_t", 0.0) > 2.0:
            j = next((idx for idx, rec in enumerate(self._meas) if not rec["done"]), None)
            if j is not None:
                print(
                    f"[M8] tick — φ̇={getattr(self, 'phi_rate', 0.0):.3f} rad/s | "
                    f"beam={j} | in={self._meas[j]['in_pts'].shape[0]} "
                    f"out={self._meas[j]['out_pts'].shape[0]}"
                )
            else:
                done_cnt = sum(1 for rec in self._meas if rec['done'])
                print(f"[M8] tick — all beams done ({done_cnt}/{self.beam_count})")
            self._last_ping_t = now

        if now - self._last_summary_t >= float(self.SUMMARY_PERIOD_S):
            self._print_validation_summary()
            self._last_summary_t = now

    def render(self):
        """Reuse Mission 6/7 pipeline (grid, BH, heads, trails)."""
        super().render()

    def handle_key(self, key, action, modifiers, keys):
        """No extra hotkeys for validation yet."""
        pass

    def _angle_in_window(self, phi, center, halfwidth):
        """Return True if angle phi is within +/- halfwidth of center (on the circle)."""
        d = (phi - center + np.pi) % (2.0*np.pi) - np.pi
        return abs(d) <= halfwidth
    # -------------------------------------------------------------------------
    # Sampling and fitting
    # -------------------------------------------------------------------------
    def _collect_far_field_samples(self):
        cx, cy = float(self.center[0]), float(self.center[1])
        r_far2 = float(self._far_radius_px) ** 2

        phi0   = float(self.phi0)
        phimax = float(self.phi_max)
        win    = float(self.PHI_WINDOW_RAD)

        # second outbound window centered at phi0 - tiny (wrap-around)
        phi_wrap_out = phi0 - 1e-3

        for i in range(self.beam_count):
            if self._meas[i]["done"]:
                continue

            x, y = float(self._positions[i, 0]), float(self._positions[i, 1])
            phi_i = float(self._phi[i])

            far_enough = ((x - cx) * (x - cx) + (y - cy) * (y - cy) >= r_far2)

            # inbound: near phi0 AND far
            if self._angle_in_window(phi_i, phi0, win/2) and far_enough:
                self._append_fit_point(i, kind="in", pt=(x, y))

            # outbound: accept if near ph_max OR near wrap-around (phi0 − ε), and far
            if (self._angle_in_window(phi_i, phimax, win/2) or
                self._angle_in_window(phi_i, phi_wrap_out, win/2)):
                if far_enough:
                    self._append_fit_point(i, kind="out", pt=(x, y))


    def _append_fit_point(self, beam_idx: int, kind: str, pt):
        """Append a single point to inbound or outbound buffer with fixed capacity."""
        key = "in_pts" if kind == "in" else "out_pts"
        arr = self._meas[beam_idx][key]
        if arr.shape[0] < int(self.FIT_SAMPLES):
            arr = np.vstack([arr, np.array(pt, dtype=np.float32)])
        else:
            arr = np.vstack([arr[1:], np.array(pt, dtype=np.float32)])
        self._meas[beam_idx][key] = arr

    def _compute_deflections_if_ready(self):
        """
        Fit unit directions along the *travel direction* (first→last sample)
        for inbound and outbound asymptotes. Deflection is |π − arccos(v_in·v_out)|.
        """
        for i in range(self.beam_count):
            rec = self._meas[i]
            if rec["done"]:
                continue
            if rec["in_pts"].shape[0] < int(self.MIN_FIT_SAMPLES):
                continue
            if rec["out_pts"].shape[0] < int(self.MIN_FIT_SAMPLES):
                continue

            v_in  = self._fit_asymptote_dir(rec["in_pts"])   # first→last
            v_out = self._fit_asymptote_dir(rec["out_pts"])  # first→last

            dot   = float(np.clip(np.dot(v_in, v_out), -1.0, 1.0))
            theta = float(np.arccos(dot))          # angle between travel directions
            delta_num = abs(np.pi - theta)         # deflection

            b_m = float(abs(self._b_m[i]))
            M   = float(self.M_geo)
            delta_ana = 4.0 * M / b_m

            error = delta_num - delta_ana
            rec["result"] = (delta_num, delta_ana, error)
            rec["done"] = True

            print(f"[M8] Beam {i:02d}: num={delta_num:.6f} rad | ana={delta_ana:.6f} rad | err={error:+.6e}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _fit_asymptote_dir(self, points_xy: np.ndarray) -> np.ndarray:
        """
        Fit a straight line using PCA (SVD) and return a unit direction vector
        oriented along the forward time direction of the samples (from first to last).
        This avoids biasing the angle by forcing 'outward-from-center' orientation.
        """
        pts = np.asarray(points_xy, dtype=np.float64)
        npts = pts.shape[0]
        if npts < 2:
            return np.array([1.0, 0.0], dtype=np.float64)

        # Center and SVD for principal axis
        mu = pts.mean(axis=0, keepdims=True)
        X = pts - mu
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        v = Vt[0, :2]  # principal direction (unit up to scale)

        # Normalize
        norm = float(np.linalg.norm(v))
        v = v / norm if norm > 0 else np.array([1.0, 0.0], dtype=np.float64)

        # Temporal orientation: ensure the direction points from first -> last sample
        delta = pts[-1] - pts[0]
        if float(np.dot(delta, v)) < 0.0:
            v = -v

        return v.astype(np.float64)

    def _fit_asymptote_dir_outward(self, points_xy: np.ndarray) -> np.ndarray:
        """
        Fit a straight line using PCA (SVD) and return a unit direction vector
        that points OUTWARD in radius (away from the black hole).
        Strategy:
        1) Get principal axis (v) via SVD.
        2) Align v with temporal direction (first → last sample).
        3) If the window is moving inward in radius (r_last < r_first),
            flip v so it points outward (opposite to temporal).
        """
        pts = np.asarray(points_xy, dtype=np.float64)
        if pts.shape[0] < 2:
            return np.array([1.0, 0.0], dtype=np.float64)

        # SVD principal axis
        mu = pts.mean(axis=0, keepdims=True)
        X  = pts - mu
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        v = Vt[0, :2]
        n = float(np.linalg.norm(v))
        v = v / n if n > 0 else np.array([1.0, 0.0], dtype=np.float64)

        # Temporal alignment (first → last)
        delta = pts[-1] - pts[0]
        if float(np.dot(delta, v)) < 0.0:
            v = -v

        # Outward orientation: if radius decreased over the window,
        # flip so the returned vector points to increasing radius.
        cx, cy = float(self.center[0]), float(self.center[1])
        r0 = float(np.hypot(pts[0, 0] - cx, pts[0, 1] - cy))
        r1 = float(np.hypot(pts[-1, 0] - cx, pts[-1, 1] - cy))
        if r1 < r0:
            v = -v

        return v.astype(np.float64)


    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    def _print_validation_summary(self):
        """
        Print a compact summary of beams with completed measurements.
        Shows mean abs error and a few sample lines.
        """
        results = [rec["result"] for rec in self._meas if rec["done"] and rec["result"] is not None]
        if not results:
            return

        errs = [abs(r[2]) for r in results]
        mean_err = float(np.mean(errs))
        max_err  = float(np.max(errs))
        count    = len(results)

        print(f"[M8] Summary: {count}/{self.beam_count} beams | mean|err|={mean_err:.3e} rad | max|err|={max_err:.3e} rad")
        shown = 0
        for i, rec in enumerate(self._meas):
            if rec["done"] and rec["result"] is not None:
                num, ana, err = rec["result"]
                print(f"      Beam {i:02d} → num={num:.6f}, ana={ana:.6f}, err={err:+.3e}")
                shown += 1
                if shown >= 3:
                    break
