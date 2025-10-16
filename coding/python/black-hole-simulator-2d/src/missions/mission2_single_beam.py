# =============================================================================
# Mission 2: Single Light Beam (Skeleton)
# =============================================================================
# This skeleton provides the structure for animating a single light beam.
# Implement the logic for particle movement, rendering, and controls.
# The vertex shader setup is provided.
# =============================================================================

# =============================================================================
# Mission 2: Single Light Beam (Skeleton)
# =============================================================================
# This skeleton provides the structure for animating a single light beam.
# Implement the logic for particle movement, rendering, and controls.
# The vertex shader setup is provided.
# =============================================================================

import numpy as np
import moderngl
from .mission1_grid_blackhole import Mission1GridBlackHole


class Mission2SingleBeam(Mission1GridBlackHole):
    """
    Mission 2: Mission 1 background (grid + black hole) + one moving light particle.
    """

    def get_name(self) -> str:
        return "Mission 2: Single Light Beam"

    def initialize(self) -> None:
        """Extend Mission 1 by adding a point-sprite particle."""
        # 1) Background from Mission 1
        super().initialize()

        # 2) Make sure the driver uses gl_PointSize from the vertex shader
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # 3) Particle shader (position given in PIXELS, we convert to NDC)
        self.pt_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;            // particle position in pixels
                uniform vec2 u_viewport;   // (width, height) in pixels
                uniform float u_point_size;

                void main() {
                    // Convert pixel space -> clip space (-1 .. +1)
                    vec2 ndc = vec2(
                        (in_pos.x / u_viewport.x) * 2.0 - 1.0,
                        (in_pos.y / u_viewport.y) * 2.0 - 1.0
                    );
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    gl_PointSize = u_point_size;
                }
            """,
            fragment_shader="""
                #version 330
                // Soft circular point
                out vec4 f_color;
                uniform vec3 u_color;

                void main() {
                    vec2 d = gl_PointCoord - vec2(0.5);
                    if (dot(d, d) > 0.25) {  // outside radius 0.5
                        discard;
                    }
                    f_color = vec4(u_color, 1.0);
                }
            """
        )

        # 4) Particle state (start slightly left of the screen center in y)
        self.beam_speed_px = 250.0
        self.point_size_px = 10.0
        self.beam_color = (1.0, 0.9, 0.4)

        # Position buffer (single vec2)
        self._particle_pos = np.array([-20.0, float(self.center[1])], dtype="f4")
        self._particle_vbo = self.ctx.buffer(self._particle_pos.tobytes())

        # VAO: feed in_pos from buffer (two floats)
        self.pt_vao = self.ctx.vertex_array(
            self.pt_prog,
            [(self._particle_vbo, "2f", "in_pos")]
        )

        # 5) Set static uniforms
        self.pt_prog["u_viewport"].value = (float(self.width), float(self.height))
        self.pt_prog["u_point_size"].value = float(self.point_size_px)
        self.pt_prog["u_color"].value = self.beam_color

        print("[M2] initialize done")

    def update(self, dt: float) -> None:
        """Move the particle right at beam_speed_px and respawn when off-screen."""
        if getattr(self, "paused", False):
            return

        self._particle_pos[0] += self.beam_speed_px * float(dt)

        # Respawn left if particle passed right edge
        if self._particle_pos[0] > self.width + 20.0:
            self._particle_pos[0] = -20.0
            self._particle_pos[1] = float(self.center[1])

        # Push updated position to GPU
        self._particle_vbo.write(self._particle_pos.tobytes())

    def render(self) -> None:
        """Draw Mission 1 background first, then the particle."""
        # Background (grid + BH)
        super().render()

        # Particle uniforms (in case size/color changed)
        self.pt_prog["u_viewport"].value = (float(self.width), float(self.height))
        self.pt_prog["u_point_size"].value = float(self.point_size_px)
        self.pt_prog["u_color"].value = self.beam_color

        # Draw single point
        self.pt_vao.render(mode=moderngl.POINTS, vertices=1)

    def handle_key(self, key, action, modifiers, keys) -> None:
        """
        Extend Mission 1 hotkeys with beam controls:
          R  : reset beam to x=-20, y=center.y
          1/2: slower / faster beam
          9/0: smaller / larger point size
          C  : cycle beam color
        """
        # Accept press + repeat for responsiveness
        allowed = {getattr(keys, "ACTION_PRESS", None), getattr(keys, "ACTION_REPEAT", None)}
        if action not in allowed:
            return

        key_name = getattr(key, "name", None) if not isinstance(key, str) else key
        key_up = key_name.upper() if isinstance(key_name, str) else None

        # Let Q/ESC close (works even without MissionControl)
        if key == getattr(keys, "ESCAPE", None) or key_up == "Q":
            self.ctx._window.close()
            return

        # Pause as in M1
        if key == getattr(keys, "SPACE", None) or key_up == "SPACE":
            self.paused = not getattr(self, "paused", False)
            return

        # Beam-specific
        if key_up == "R":
            self._particle_pos[:] = (-20.0, float(self.center[1]))
            self._particle_vbo.write(self._particle_pos.tobytes())
            return

        if key_up == "1":
            self.beam_speed_px = max(10.0, self.beam_speed_px / 1.2); return
        if key_up == "2":
            self.beam_speed_px = min(2000.0, self.beam_speed_px * 1.2); return

        if key_up == "9":
            self.point_size_px = max(2.0, self.point_size_px / 1.2); return
        if key_up == "0":
            self.point_size_px = min(64.0, self.point_size_px * 1.2); return

        if key_up == "C":
            palette = [
                (1.0, 0.9, 0.4),
                (0.4, 0.8, 1.0),
                (1.0, 0.4, 0.4),
                (0.7, 1.0, 0.7),
            ]
            try:
                idx = next(i for i, col in enumerate(palette) if col == self.beam_color)
                self.beam_color = palette[(idx + 1) % len(palette)]
            except StopIteration:
                self.beam_color = palette[0]
            return

        # Fall back to Mission 1 controls (move BH, change radius, grid, etc.)
        super().handle_key(key, action, modifiers, keys)
