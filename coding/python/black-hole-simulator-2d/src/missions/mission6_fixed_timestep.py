# =============================================================================
# Mission 6: Fixed Timestep & Ray Trails (Skeleton)
# =============================================================================
# -----------------------------------------------------------------------------
# What we add on top of Mission 5:
#   • A fixed-timestep loop (dt accumulator) for stable & deterministic updates.
#   • A per-beam trail implemented as a circular buffer of past positions.
#   • A dedicated trail shader (points + alpha fade with age).
#
# Why fixed timestep?
#   Rendering (display refresh) and physics (state evolution) have different needs.
#   A "variable dt" integrator (advance by the render frame's dt) can jitter and
#   behave differently on fast vs slow machines. A fixed timestep (e.g. 1/240 s)
#   decouples physics from framerate: you take 0..N *fixed* substeps per frame.
#
# Data layout for trails:
#   trail_positions : shape (trail_len, beam_count, 2)   -- in pixels
#   trail_head      : int index (where the next sample will be written)
#   During render we flatten starting at the oldest sample (trail_head) so that
#   vertex 0 has age=oldest, vertex ... has age=newest (age in [0..1]).
# =============================================================================

import numpy as np
import moderngl
from .mission5_units_schwarzschild import Mission5UnitsSchwarzschild

class Mission6FixedTimestep(Mission5UnitsSchwarzschild):
    """Rays with fixed timestep integration and nice trails."""

    def get_name(self):
        return "Mission 6: Rays with Trails (Fixed Timestep)"

    # -------------------------------------------------------------------------
    # 1) Initialization
    # -------------------------------------------------------------------------
    def initialize(self):
        """
        Build everything from Mission 5 (background, beams, per-vertex-color heads,
        SI conversions), then allocate trail buffers and the trail shader.
        """
        super().initialize()

        # --- Make sure the GPU will respect gl_PointSize in shaders -------------
        # Some drivers require this to be on for point sprites to size correctly.
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # --- Fixed timestep parameters ------------------------------------------
        self.fixed_dt_s: float = 1.0 / 240.0  # physics substep
        self._accum_s: float = 0.0            # leftover time accumulator
        self.max_substeps: int = 8            # safety cap per frame

        # --- Ensure previous-mission fields exist -------------------------------
        # Heads of rays: shape (N, 2) in pixels. Also need beam_count.
        if not hasattr(self, "_positions"):
            # Fallback: create a tiny set of rays if previous missions didn't.
            import numpy as np
            self.beam_count = 16
            x0 = float(getattr(self, "left_margin_px", 0.0)) if hasattr(self, "left_margin_px") else 10.0
            y0 = float(self.center[1])
            gap = float(getattr(self, "beam_spacing_px", 18.0))
            ys = y0 + (np.arange(self.beam_count, dtype=np.float32) - (self.beam_count - 1) * 0.5) * gap
            xs = np.full_like(ys, x0, dtype=np.float32)
            self._positions = np.column_stack([xs, ys]).astype("f4")
        else:
            self.beam_count = int(self._positions.shape[0])

        # Default point size if Mission 2 didn't set it
        if not hasattr(self, "point_size_px"):
            self.point_size_px = 6.0

        # --- Trail configuration -------------------------------------------------
        self.trail_enabled: bool = True
        self.trail_len: int = 64                                   # samples per beam
        self.trail_point_size_px: float = max(2.0, float(self.point_size_px) * 0.6)
        
        self.trail_head: int = 0                                   # circular write index

        # Allocate trail ring buffer in CPU memory and prefill with current heads.
        # Shape: (trail_len, beam_count, 2) in float32 (pixel space).
        import numpy as np
        self.trail_positions = np.repeat(self._positions[None, :, :], self.trail_len, axis=0).astype("f4")
        # Ages (0=newest .. 1=oldest) per sample row; same for all beams initially.
        # We'll expand to per-vertex in the VBO.
        base_age = np.linspace(1.0, 0.0, self.trail_len, dtype=np.float32)

        # --- Trail shader --------------------------------------------------------
        self.trail_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2  in_pos;       // pixel coordinates
                in float in_age;       // 0=newest ... 1=oldest
                out float v_age;

                uniform vec2  u_viewport;
                uniform float u_point_size;

                void main() {
                    // Convert pixels → NDC
                    vec2 ndc = vec2(
                        (in_pos.x / u_viewport.x) * 2.0 - 1.0,
                        (in_pos.y / u_viewport.y) * 2.0 - 1.0
                    );
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    gl_PointSize = u_point_size;
                    v_age = in_age;
                }
            """,
            fragment_shader="""
                #version 330
                in float v_age;
                out vec4 f_color;

                uniform vec3 u_color;   // base trail color (usually beam_color)
                uniform float u_gamma;  // fade exponent (1.0 .. 2.5)

                void main() {
                    // Circular point sprite
                    vec2 d = gl_PointCoord - vec2(0.5);
                    if (dot(d, d) > 0.25) discard;

                    // Newest (0) opaque, oldest (1) transparent
                    float alpha = pow(1.0 - clamp(v_age, 0.0, 1.0), u_gamma);
                    f_color = vec4(u_color, alpha);
                }
            """,
        )

        # --- GPU buffers for trail data -----------------------------------------
        # We will stream (positions, age) every frame. Compute total vertices.
        nverts = int(self.trail_len * self.beam_count)
        # Each pos = 2 * float32 (8 bytes), each age = 1 * float32 (4 bytes).
        self._trail_pos_vbo = self.ctx.buffer(reserve=nverts * 2 * 4)
        self._trail_age_vbo = self.ctx.buffer(reserve=nverts * 1 * 4)

        self._trail_vao = self.ctx.vertex_array(
            self.trail_prog,
            [
                (self._trail_pos_vbo, "2f", "in_pos"),
                (self._trail_age_vbo, "f",  "in_age"),
            ],
        )

        # --- Static uniforms (also updated in render) ---------------------------
        self.trail_prog["u_viewport"].value = (float(self.width), float(self.height))
        self.trail_prog["u_point_size"].value = float(self.trail_point_size_px)
        self.trail_prog["u_gamma"].value = 1.6

        # --- Initial write of VBO contents --------------------------------------
        # Flatten CPU buffers into interleaved vertex order: for each time slice, all beams.
        def _initial_trail_upload():
            # positions: (L,N,2) -> (L*N,2)
            pos = np.ascontiguousarray(self.trail_positions.reshape(-1, 2))
            self._trail_pos_vbo.write(pos.tobytes())

            # ages: repeat each age for all beams -> (L,N) -> (L*N,)
            ages = np.repeat(base_age[:, None], self.beam_count, axis=1).reshape(-1).astype("f4")
            self._trail_age_vbo.write(ages.tobytes())

        _initial_trail_upload()

    # -------------------------------------------------------------------------
    # 2) Update with FIXED TIMESTEP + trail sampling
    # -------------------------------------------------------------------------
    def update(self, dt):
        """
        Accumulate render dt, take k = floor(accum / fixed_dt) substeps (capped),
        and after *each* substep, store the current beam heads into the trail ring.
        Then write flattened buffers for rendering.
        """
        if getattr(self, "paused", False):
            return

        # Accumulate time
        self._accum_s += float(dt)

        # Determine number of fixed substeps to execute this frame
        steps_wanted = int(self._accum_s / self.fixed_dt_s)
        steps = min(steps_wanted, self.max_substeps)

        for _ in range(steps):
            # Move beams and do collisions with a *fixed* dt step.
            # Mission 5 update also recomputes SI → pixels each call so keys
            # (mass, zoom, speed) react immediately.
            super().update(self.fixed_dt_s)

            # Record heads into the ring buffer at the current trail_head
            self.trail_positions[self.trail_head, :, :] = self._positions
            self.trail_head = (self.trail_head + 1) % self.trail_len

            # Consume time
            self._accum_s -= self.fixed_dt_s

        # Prepare GPU buffers (flatten oldest→newest with matching age)
        # Oldest sample index is exactly trail_head (next write slot).
        order = (np.arange(self.trail_len, dtype=np.int32) + self.trail_head) % self.trail_len
        flat_pos = self.trail_positions[order, :, :].reshape(-1, 2)

        # Ages in [0..1]: 0=newest (last), 1=oldest (first).
        # Since we arranged oldest→newest, age grows linearly from 1→0:
        if self.trail_len > 1:
            ages_line = np.linspace(1.0, 0.0, self.trail_len, dtype="f4")
        else:
            ages_line = np.array([0.0], dtype="f4")
        flat_age = np.repeat(ages_line[:, None], self.beam_count, axis=1).reshape(-1)

        # Push to GPU
        self._trail_pos_vbo.write(flat_pos.astype("f4").tobytes())
        self._trail_age_vbo.write(flat_age.tobytes())

    # -------------------------------------------------------------------------
    # 3) Render: trails → heads (parent) → done
    # -------------------------------------------------------------------------
    def render(self):
        """
        We draw trails *on top* of whatever parent draws. This keeps the code simple
        (we don't split the parent's background vs heads). Blending is enabled
        in MissionControl, so faded points look correct.
        """
        # First let parent draw background + heads
        super().render()

        if not self.trail_enabled:
            return

        # Then draw trails as semi-transparent points
        self.trail_prog["u_viewport"].value = (float(self.width), float(self.height))
        self.trail_prog["u_point_size"].value = float(self.trail_point_size_px)
        # Trail color: re-use the current beam base color, but you can tint it
        self.trail_prog["u_color"].value = tuple(float(v) for v in getattr(self, "beam_color", (1.0, 0.9, 0.4)))

        nverts = int(self.trail_len * self.beam_count)
        if nverts > 0:
            self._trail_vao.render(mode=moderngl.POINTS, vertices=nverts)

    # -------------------------------------------------------------------------
    # 4) Keys (also callable from MissionControl polling)
    # -------------------------------------------------------------------------
    def toggle_trails(self) -> None:
        """Toggle drawing trails on/off."""
        self.trail_enabled = not getattr(self, "trail_enabled", True)
        print(f"[M6] Trails: {'ON' if self.trail_enabled else 'OFF'}")

    def decrease_trail_len(self) -> None:
        """Halve trail length (min 4) preserving newest samples."""
        new_len = max(4, int(getattr(self, "trail_len", 64) / 2))
        if new_len != getattr(self, "trail_len", 64):
            self._resize_trail(new_len)
            print(f"[M6] Trail length: {self.trail_len}")

    def increase_trail_len(self) -> None:
        """Increase trail length by ×1.5 up to a cap (512)."""
        cur = int(getattr(self, "trail_len", 64))
        new_len = min(512, int(cur * 1.5))
        if new_len != cur:
            self._resize_trail(new_len)
            print(f"[M6] Trail length: {self.trail_len}")

    def handle_key(self, key, action, modifiers, keys):
        pass

    # ---------- Mission 6: trail helpers & hotkey API ----------

    def _resize_trail(self, new_len: int):
        """
        Reallocate the trail ring buffer and GPU buffers to 'new_len' samples per beam.
        - Preserves as much recent history as possible.
        - Rebuilds VBOs and the VAO used for trail rendering.
        - Keeps the newest sample at the end of the ring (head = new_len - 1).
        """

        # 1) Clamp and short-circuit
        new_len = int(max(8, min(int(new_len), 4096)))
        if getattr(self, "trail_len", None) == new_len:
            return

        if not hasattr(self, "beam_count") or not hasattr(self, "_positions"):
            # Nothing to do until Mission 5/6 have created beams/heads.
            print("[M6] _resize_trail skipped (no beams yet)")
            return

        old_len = int(getattr(self, "trail_len", new_len))
        beam_count = int(self.beam_count)

        # 2) Prepare new CPU ring buffer
        new_trail = np.empty((new_len, beam_count, 2), dtype="f4")

        if hasattr(self, "trail_positions") and old_len > 0:
            # Copy the most-recent min(old_len, new_len) samples preserving order (oldest..newest)
            k = min(old_len, new_len)
            old = self.trail_positions
            head = int(getattr(self, "trail_head", old_len - 1)) % old_len

            # Fill with current heads by default (for the part we can't preserve)
            heads_now = self._positions.astype("f4")
            for i in range(new_len):
                new_trail[i, :, :] = heads_now  # will be overwritten below for preserved range

            # Copy last k samples: dst [new_len - k .. new_len-1] becomes oldest..newest
            for j in range(k):
                src_idx = (head - (k - 1 - j)) % old_len   # walk from oldest to newest
                dst_idx = new_len - k + j
                new_trail[dst_idx, :, :] = old[src_idx, :, :]
        else:
            # No previous history: seed all rows with current heads
            heads_now = self._positions.astype("f4")
            new_trail[:] = heads_now[None, :, :]

        # 3) Swap CPU buffers and indices
        self.trail_positions = new_trail
        self.trail_len = new_len

        self.trail_head = 0  # invariant: head == next write; oldest == head

        # 4) Recreate GPU buffers sized for new_len * beam_count
        nverts = int(new_len * beam_count)

        # Release old buffers if they exist (moderngl handles GC, но явный release — аккуратнее)
        if hasattr(self, "_trail_pos_vbo"):
            try:
                self._trail_pos_vbo.release()
            except Exception:
                pass
        if hasattr(self, "_trail_age_vbo"):
            try:
                self._trail_age_vbo.release()
            except Exception:
                pass
        if hasattr(self, "_trail_vao"):
            try:
                self._trail_vao.release()
            except Exception:
                pass

        self._trail_pos_vbo = self.ctx.buffer(reserve=nverts * 2 * 4)  # 2 floats per vertex
        self._trail_age_vbo = self.ctx.buffer(reserve=nverts * 1 * 4)  # 1 float per vertex

        # Rebuild VAO with the existing trail shader program
        self._trail_vao = self.ctx.vertex_array(
            self.trail_prog,
            [
                (self._trail_pos_vbo, "2f", "in_pos"),
                (self._trail_age_vbo, "f",  "in_age"),
            ],
        )

        # 5) Initial upload (oldest..newest order) and ages in [0..1]
        # Flatten (row-major): rows are chronological (oldest first), then beams
        pos_cpu = self.trail_positions.reshape(-1, 2)
        self._trail_pos_vbo.write(pos_cpu.tobytes())

        if new_len > 1:
            ages_row = np.linspace(1.0, 0.0, new_len, dtype="f4")
        else:
            ages_row = np.array([0.0], dtype="f4")
        ages = np.repeat(ages_row, beam_count).astype("f4")
        self._trail_age_vbo.write(ages.tobytes())

        # 6) Keep uniforms in sync (viewport/point size/gamma)
        if hasattr(self, "trail_prog"):
            self.trail_prog["u_viewport"].value = (float(self.width), float(self.height))
            self.trail_prog["u_point_size"].value = float(getattr(self, "trail_point_size_px", 4.0))
            self.trail_prog["u_gamma"].value = float(getattr(self, "trail_gamma", 1.6))

        print(f"[M6] trail resized → {self.trail_len} samples/beam (nverts={nverts})")


    def clear_trails(self):
        """
        Clear history: fill the ring buffer with the current head positions.
        Does not change the trail length.
        """
        import numpy as np

        if not hasattr(self, "trail_positions"):
            return
        heads_now = self._positions.astype("f4")
        self.trail_positions[:] = heads_now[None, :, :]
        self.trail_head = 0

        # Upload to GPU immediately to reflect cleared state
        pos_cpu = self.trail_positions.reshape(-1, 2)
        self._trail_pos_vbo.write(pos_cpu.tobytes())

        if self.trail_len > 1:
            ages_row = np.linspace(1.0, 0.0, int(self.trail_len), dtype="f4")
        else:
            ages_row = np.array([0.0], dtype="f4")
        ages = np.repeat(ages_row, int(self.beam_count)).astype("f4")
        self._trail_age_vbo.write(ages.tobytes())

        print("[M6] trails cleared")
