# =============================================================================
# Mission 1: Grid Background and Black Hole Disc (Skeleton)
# =============================================================================
# This skeleton provides the OpenGL vertex shader setup for rendering a grid and black hole disc.
# Implement the logic for initializing, updating, and rendering the mission.
# =============================================================================

import numpy as np
import moderngl
from .base_mission import BaseMission


class Mission1GridBlackHole(BaseMission):
    """
    Mission 1: Render a pixel-aligned grid and a filled black-hole disc (in pixel units).

    Key ideas:
    - We draw a single full-screen triangle (no vertex buffers needed).
    - The fragment shader paints each pixel using gl_FragCoord (actual pixel coords).
    - We control the disc center and radius via uniforms (u_center, u_rsPx).
    - We control grid spacing (in pixels) via uniform (u_grid_gap).
    """

    def get_name(self) -> str:
        return "Mission 1: Grid + Black Hole"

    def initialize(self):
        """
        Build the GPU program (vertex + fragment), create an empty VAO,
        and initialize uniforms/state.

        Why full-screen triangle?
        - It's the simplest way to cover the entire viewport without creating a VBO.
        - The vertex shader uses gl_VertexID (0,1,2) to construct positions that
          form a giant triangle covering the whole screen after rasterization.
        """
        # --- 1) Shader program: vertex + fragment ---------------------------
        self.bg_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            // Full-screen triangle via gl_VertexID.
            //   id=0 -> (-1, -1)
            //   id=1 -> ( 3, -1)
            //   id=2 -> (-1,  3)
            // This single triangle fully covers the screen.
            out vec2 v_uv;  // optional UV in [0,1] (not required for pixel grid)

            void main() {
                float x = -1.0 + float((gl_VertexID & 1) << 2);
                float y = -1.0 + float((gl_VertexID & 2) << 1);
                v_uv = vec2(x, y) * 0.5 + 0.5;   // keep for future effects if needed
                gl_Position = vec4(x, y, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            // Paint a crisp 1-pixel grid and a filled black-hole disc in pixel space.
            // We use gl_FragCoord (pixel coordinates) for exact grid alignment.
            in vec2 v_uv;
            out vec4 f_color;

            // Uniforms set from CPU (Python) each frame:
            uniform vec2  u_center;    // black hole center in pixels (x,y)
            uniform float u_rsPx;      // black hole radius in pixels
            uniform float u_grid_gap;  // grid spacing in pixels (distance between lines)

            void main() {
                // Pixel position of this fragment (1-based in OpenGL).
                vec2 frag = vec2(gl_FragCoord.x, gl_FragCoord.y);

                // --- Grid: draw a line when modulo < 1 pixel (crisp 1px lines)
                float gx = (mod(frag.x, u_grid_gap) < 1.0) ? 1.0 : 0.0;
                float gy = (mod(frag.y, u_grid_gap) < 1.0) ? 1.0 : 0.0;
                float grid = max(gx, gy);

                // Background and grid colors (tweak to taste)
                vec3 bg       = vec3(0.03, 0.04, 0.07);
                vec3 grid_col = vec3(0.12, 0.14, 0.20);

                // Base color with grid overlay
                vec3 col = mix(bg, grid_col, grid);

                // --- Black-hole disc: fill all pixels within radius u_rsPx
                float r = length(frag - u_center);
                if (r <= u_rsPx) {
                    col = vec3(0.0);  // solid black disc
                }

                f_color = vec4(col, 1.0);
            }
            """
        )

        # --- 2) VAO (Vertex Array Object) ----------------------------------
        # No vertex buffers needed since the vertex shader uses gl_VertexID.
        self.vao = self.ctx.vertex_array(self.bg_prog, [])

        # --- 3) Mission state ----------------------------------------------
        # Use BaseMission-provided fields:
        #   self.center : np.ndarray([x,y]) in pixels (defaults to screen center)
        #   self.rs_px  : float radius in pixels (defaults to 12% of min(width, height))
        self.grid_gap_px: float = 32.0  # 32px between grid lines
        self.paused: bool = False       # reserved for future animations in update()

        # --- 4) Initialize uniforms once (will also be updated every render)-
        self.bg_prog["u_center"].value   = (float(self.center[0]), float(self.center[1]))
        self.bg_prog["u_rsPx"].value     = float(self.rs_px)
        self.bg_prog["u_grid_gap"].value = float(self.grid_gap_px)

    def update(self, dt: float):
        """
        Advance simulation by dt seconds (called each frame).

        Mission 1 does not require animation, so we keep it simple.
        If you want a learning exercise, try enabling a subtle "breathing"
        effect by modulating self.rs_px over time (example commented out).
        """
        if self.paused:
            return

        # Example (uncomment to experiment):
        # import math
        # base = float(min(self.width, self.height) * 0.12)
        # self.rs_px = base * (1.0 + 0.03 * math.sin(0.8 * time_accumulator))
        # Where `time_accumulator` would be incremented by dt somewhere you store it.

    def render(self):
        """
        Send current CPU state to the GPU and draw one full-screen triangle.

        MissionControl already clears the screen every frame:
        self.ctx.clear(0.03, 0.04, 0.07, 1.0)
        So we just push uniforms and render.
        """
        # Push uniforms derived from BaseMission fields:
        self.bg_prog["u_center"].value   = (float(self.center[0]), float(self.center[1]))
        self.bg_prog["u_rsPx"].value     = float(self.rs_px)
        self.bg_prog["u_grid_gap"].value = float(self.grid_gap_px)

        # Draw 3 vertices (one full-screen triangle)
        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)

    def handle_key(self, key, action, modifiers, keys):
        """
        Robust keyboard handler for Mission 1.
        Supports:
        - SPACE            : pause / unpause
        - = / + / KP_ADD   : increase radius
        - - / KP_SUBTRACT  : decrease radius
        - [ / ]            : grid spacing down/up
        - Arrows or WASD   : move center
        - Q or ESC         : quit (close window)
        """
        # --- DEBUG first line: prove we're called and see key names ---
        key_name = getattr(key, "name", None) if not isinstance(key, str) else key
        key_up = key_name.upper() if isinstance(key_name, str) else None
        print(f"[M1] key={key} name={key_name} action={action} mods={modifiers}")

        # Accept PRESS and REPEAT
        allowed = {getattr(keys, "ACTION_PRESS", None), getattr(keys, "ACTION_REPEAT", None)}
        if action not in allowed:
            return

        # Quit
        if key == getattr(keys, "ESCAPE", None) or key_up == "Q":
            self.ctx._window.close()
            return

        # Pause
        if key == getattr(keys, "SPACE", None) or key_up == "SPACE":
            self.paused = not getattr(self, "paused", False)
            return

        # Radius increase
        if (
            key == getattr(keys, "EQUAL", None) or
            key == getattr(keys, "KP_ADD", None) or
            key_up in ("EQUAL", "PLUS", "+", "KP_ADD")
        ):
            self.rs_px *= 1.1
            return

        # Radius decrease
        if (
            key == getattr(keys, "MINUS", None) or
            key == getattr(keys, "KP_SUBTRACT", None) or
            key_up in ("MINUS", "-", "KP_SUBTRACT")
        ):
            self.rs_px /= 1.1
            return

        # Move center
        step = 10.0
        if key in (getattr(keys, "UP", None), getattr(keys, "W", None)) or key_up in ("UP", "W"):
            self.center[1] = min(self.height, self.center[1] + step); return
        if key in (getattr(keys, "DOWN", None), getattr(keys, "S", None)) or key_up in ("DOWN", "S"):
            self.center[1] = max(0.0, self.center[1] - step); return
        if key in (getattr(keys, "LEFT", None), getattr(keys, "A", None)) or key_up in ("LEFT", "A"):
            self.center[0] = max(0.0, self.center[0] - step); return
        if key in (getattr(keys, "RIGHT", None), getattr(keys, "D", None)) or key_up in ("RIGHT", "D"):
            self.center[0] = min(self.width, self.center[0] + step); return

        # Grid spacing
        if key == getattr(keys, "LEFT_BRACKET", None) or key_name == "[":
            self.grid_gap_px = max(4.0, getattr(self, "grid_gap_px", 32.0) / 1.2); return
        if key == getattr(keys, "RIGHT_BRACKET", None) or key_name == "]":
            self.grid_gap_px = min(256.0, getattr(self, "grid_gap_px", 32.0) * 1.2); return
