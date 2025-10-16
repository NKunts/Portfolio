# =============================================================================
# Mission 4: Multiple Light Beams with Collision Detection (Skeleton)
# =============================================================================
# This skeleton provides the structure for managing multiple light beams and adding collision detection.
# Implement the logic for detecting collisions with the black hole and updating beam states.
# The vertex shader setup from previous missions is available for reuse.
# =============================================================================


import numpy as np
import moderngl
from .mission3_multiple_beams_no_collision import Mission3MultipleBeamsNoCollision


class Mission4MultipleBeams(Mission3MultipleBeamsNoCollision):
    """
    Mission 4: Multiple beams + collision detection against the black hole disc.
    - Reuse background, array-of-positions, timing from previous missions.
    - Add per-beam color so we can highlight collisions.
    - On collision: flash the beam (short color pulse). Optionally respawn.
    """

    def get_name(self) -> str:
        return "Mission 4: Multiple Light Beams"

    # ---------------------------------------------------------------------
    # 1) Initialization
    # ---------------------------------------------------------------------
    def initialize(self) -> None:
        """
        Build on Mission 3:
          - Call super().initialize() to set up positions, VBO, spacing, etc.
          - Replace the point-sprite program with a per-vertex color version.
          - Add a color buffer and a short 'flash' timer per beam.
        """
        super().initialize()  # builds positions, _vbo, _vao (we will override VAO)

        # Ensure point size is honored by the driver (already enabled in M2, but ensure again)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # --- New program: per-vertex color (in_color) ---------------------
        self.pt4_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                // Per-vertex position in pixels and color (rgb)
                in vec2 in_pos;
                in vec3 in_color;

                out vec3 v_color;

                uniform vec2 u_viewport;   // (width, height) in pixels
                uniform float u_point_size;

                void main() {
                    vec2 ndc = vec2(
                        (in_pos.x / u_viewport.x) * 2.0 - 1.0,
                        (in_pos.y / u_viewport.y) * 2.0 - 1.0
                    );
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    gl_PointSize = u_point_size;
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 f_color;

                // Round point using gl_PointCoord
                void main() {
                    vec2 d = gl_PointCoord - vec2(0.5);
                    if (dot(d, d) > 0.25) {
                        discard;
                    }
                    f_color = vec4(v_color, 1.0);
                }
            """
        )

        # --- Per-beam state ------------------------------------------------
        # Colors per beam (default = beam_color from M2)
        base_col = np.array(self.beam_color, dtype="f4")
        self._colors = np.repeat(base_col[None, :], self.beam_count, axis=0).astype("f4")

        # Flash timers (seconds). When >0, draw collision color instead of base.
        self._flash = np.zeros((self.beam_count,), dtype="f4")

        # Collision statistics / switches
        self.collision_count = 0
        self.respawn_on_hit = True           # if True: beam re-enters from left after hit
        self.collision_flash_time = 0.18     # seconds

        # --- GPU buffers / VAO --------------------------------------------
        self._cbo = self.ctx.buffer(self._colors.tobytes())  # color buffer object

        # New VAO uses position + color buffers with the new shader
        self._vao = self.ctx.vertex_array(
            self.pt4_prog,
            [
                (self._vbo, "2f", "in_pos"),
                (self._cbo, "3f", "in_color"),
            ],
        )

        # Static uniforms
        self.pt4_prog["u_viewport"].value = (float(self.width), float(self.height))
        self.pt4_prog["u_point_size"].value = float(self.point_size_px)

    # ---------------------------------------------------------------------
    # 2) Update (move + detect collisions + flash logic)
    # ---------------------------------------------------------------------
    def update(self, dt: float) -> None:
        """
        - Move active beams in +x.
        - Detect hits: if distance(center, beam) <= rs_px.
        - On hit: increment stats, start flash timer, and either respawn immediately or freeze.
        - Update per-beam colors according to flash timers.
        """
        if getattr(self, "paused", False):
            return

        # 2.1 Move beams (vectorized)
        self._positions[:, 0] += float(self.beam_speed_px) * float(dt)

        # 2.2 Respawn beams that passed the right margin (as in M3)
        passed = self._positions[:, 0] > self.right_limit_px
        if np.any(passed):
            self._positions[passed, 0] = self.left_margin_px

        # 2.3 Collision detection against BH disc (pixel units)
        cx, cy = float(self.center[0]), float(self.center[1])
        dx = self._positions[:, 0] - cx
        dy = self._positions[:, 1] - cy
        r2 = dx * dx + dy * dy
        hit_mask = r2 <= float(self.rs_px) * float(self.rs_px)

        # 2.4 Handle hits
        if np.any(hit_mask):
            self.collision_count += int(hit_mask.sum())
            # start flash timers
            self._flash[hit_mask] = self.collision_flash_time

            if self.respawn_on_hit:
                # Put hit beams back to the left edge (keep their y)
                self._positions[hit_mask, 0] = self.left_margin_px
            else:
                # Freeze at current x (optional mode): clamp x so they no longer move
                # Here we just zero their x velocity by undoing this frame's displacement:
                self._positions[hit_mask, 0] -= float(self.beam_speed_px) * float(dt)

        # 2.5 Update flash timers
        if np.any(self._flash > 0.0):
            self._flash -= float(dt)
            self._flash = np.maximum(self._flash, 0.0)

        # 2.6 Update per-vertex colors (flash color for beams with flash>0)
        # Collision flash color (reddish)
        hit_color = np.array([1.0, 0.3, 0.3], dtype="f4")
        base_col = np.array(self.beam_color, dtype="f4")

        # If flashing -> hit_color; else -> base
        flashing = self._flash > 0.0
        if np.any(flashing):
            self._colors[flashing, :] = hit_color
        if np.any(~flashing):
            self._colors[~flashing, :] = base_col

        # 2.7 Push updates to GPU
        self._vbo.write(self._positions.tobytes())
        self._cbo.write(self._colors.tobytes())

    # ---------------------------------------------------------------------
    # 3) Render (background + colored points)
    # ---------------------------------------------------------------------
    def render(self) -> None:
        # Background (grid + BH)
        super().render()

        # Shader uniforms
        self.pt4_prog["u_viewport"].value = (float(self.width), float(self.height))
        self.pt4_prog["u_point_size"].value = float(self.point_size_px)

        # Draw N points with per-vertex colors
        self._vao.render(mode=moderngl.POINTS, vertices=self.beam_count)

    # ---------------------------------------------------------------------
    # 4) Input (augment M3)
    # ---------------------------------------------------------------------
    def handle_key(self, key, action, modifiers, keys) -> None:
        """
        Extend M3 hotkeys:
          - T : toggle respawn_on_hit (respawn vs. freeze-on-hit)
          - H : clear flash & reset hit counter (re-arm all beams)
          (R, ',' and '.' remain from M3; 1/2 speed, 9/0 size, C color, etc.)
        """
        allowed = {getattr(keys, "ACTION_PRESS", None), getattr(keys, "ACTION_REPEAT", None)}
        if action not in allowed:
            return

        key_name = getattr(key, "name", None) if not isinstance(key, str) else key
        up = key_name.upper() if isinstance(key_name, str) else None

        if up == "T":
            self.respawn_on_hit = not self.respawn_on_hit
            print(f"[M4] respawn_on_hit = {self.respawn_on_hit}")
            return

        if up == "H":
            # Clear flash and counter; re-color to base
            self._flash[:] = 0.0
            base_col = np.array(self.beam_color, dtype="f4")
            self._colors[:] = base_col
            self._cbo.write(self._colors.tobytes())
            self.collision_count = 0
            print("[M4] hits cleared")
            return

        # Defer everything else to M3/M2 (spacing, reset, speed/size/color, pause/quit)
        super().handle_key(key, action, modifiers, keys)

    # Utility (already in M3, but we keep it explicit for clarity):
    def reset_all_beams(self) -> None:
        offsets = (np.arange(self.beam_count, dtype=np.float32)
                   - (self.beam_count - 1) * 0.5) * self.beam_spacing_px
        self._positions[:, 0] = self.left_margin_px
        self._positions[:, 1] = float(self.center[1]) + offsets
        self._vbo.write(self._positions.tobytes())

        # Utility methods for toggling respawn and clearing hits
    def toggle_respawn(self):
        self.respawn_on_hit = not self.respawn_on_hit
        print(f"[M4] respawn_on_hit = {self.respawn_on_hit}")

    def clear_hits(self):
        # Clear flash and counter; re-color to base
        self._flash[:] = 0.0
        base_col = np.array(self.beam_color, dtype="f4")
        self._colors[:] = base_col
        self._cbo.write(self._colors.tobytes())
        self.collision_count = 0
        print("[M4] hits cleared")

