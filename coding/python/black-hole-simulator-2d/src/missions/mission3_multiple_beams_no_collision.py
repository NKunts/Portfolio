# =============================================================================
# Mission 3: Multiple Light Beams (No Collision) Skeleton
# =============================================================================
# This skeleton provides the structure for managing and rendering multiple light beams.
# Implement the logic for spawning, updating, and rendering multiple beams.
# The vertex shader setup from previous missions is available for reuse.
# =============================================================================


# =============================================================================
# Mission 3: Multiple Light Beams (No Collision)
# - Reuse Mission 1 background (grid + BH) and Mission 2 point-sprite shader.
# - Manage an array of beam positions in pixel space and render them as GL_POINTS.
# - Animate all beams left->right with respawn when exiting the right edge.
# - Provide simple controls to reset beams and adjust spacing.
# =============================================================================

# =============================================================================
# Mission 3: Multiple Light Beams (No Collision)
# - Reuse Mission 1 background (grid + BH) and Mission 2 point-sprite shader.
# - Manage an array of beam positions in pixel space and render them as GL_POINTS.
# - Animate all beams left->right with respawn when exiting the right edge.
# - Provide simple controls to reset beams and adjust spacing.
# =============================================================================

import numpy as np
import moderngl
from .mission2_single_beam import Mission2SingleBeam


class Mission3MultipleBeamsNoCollision(Mission2SingleBeam):
    """
    Mission 3: Multiple parallel light beams rendered as point sprites.
    No collision detection yet (that comes later).
    """

    def get_name(self) -> str:
        return "Mission 3: Multiple Light Beams"

    # ---------------------------------------------------------------------
    # 1) Initialization
    # ---------------------------------------------------------------------
    def initialize(self) -> None:
        """
        Build on top of Missions 1 & 2:
          - super().initialize() sets up the background and creates self.pt_prog
            (the point-sprite shader), enables PROGRAM_POINT_SIZE, etc.
          - Here we allocate a positions array for N beams and make one VBO/VAO.
        """
        # Initialize background + point shader (from Missions 1 & 2)
        super().initialize()

        # --- Beam field parameters ---------------------------------------
        self.beam_count: int = 24          # number of beams (try 8..200)
        self.beam_spacing_px: float = 16.0 # vertical distance between beams
        self.left_margin_px: float = -20.0 # respawn x (just left of screen)
        self.right_limit_px: float = float(self.width) + 20.0  # off-screen

        # Create a regularly spaced set of y positions centered on BH center
        # Example: for 5 beams with spacing 16, offsets would be -32,-16,0,16,32
        offsets = (np.arange(self.beam_count, dtype=np.float32)
                   - (self.beam_count - 1) * 0.5) * self.beam_spacing_px
        y_center = float(self.center[1])
        y_vals = y_center + offsets

        # Start all beams slightly left of screen so they fly in
        x_vals = np.full(self.beam_count, self.left_margin_px, dtype=np.float32)

        # Positions array (N x 2) in *pixels*
        self._positions = np.column_stack([x_vals, y_vals]).astype("f4")

        # Single VBO holding all beam positions (stream updated each frame)
        self._vbo = self.ctx.buffer(self._positions.tobytes())

        # VAO: feed "in_pos" (2 floats) from buffer to the point-sprite vertex shader
        self._vao = self.ctx.vertex_array(
            self.pt_prog,
            [(self._vbo, "2f", "in_pos")]
        )

        # Keep using Mission 2 uniforms (viewport/size/color set in render())
        # Speed/size/color already exist from Mission 2 and work the same way.

    # ---------------------------------------------------------------------
    # 2) Update (animation)
    # ---------------------------------------------------------------------
    def update(self, dt: float) -> None:
        """
        Move all beams in +x by beam_speed_px * dt.
        When a beam exits to the right, respawn it at left_margin_px keeping its y.
        """
        if getattr(self, "paused", False):
            return

        # Vectorized move in x
        self._positions[:, 0] += float(self.beam_speed_px) * float(dt)

        # Respawn off-screen beams to the left; keep their current y
        mask = self._positions[:, 0] > self.right_limit_px
        if np.any(mask):
            self._positions[mask, 0] = self.left_margin_px

        # Push CPU positions to GPU
        self._vbo.write(self._positions.tobytes())

    # ---------------------------------------------------------------------
    # 3) Render (background + all beams)
    # ---------------------------------------------------------------------
    def render(self) -> None:
        """
        Draw the Mission 1 background, then render N point sprites in one call.
        """
        # Background (grid + BH disc)
        super().render()

        # Update uniforms in case size/color changed via hotkeys
        self.pt_prog["u_viewport"].value = (float(self.width), float(self.height))
        self.pt_prog["u_point_size"].value = float(self.point_size_px)
        self.pt_prog["u_color"].value = self.beam_color

        # Draw all beams as GL_POINTS (N vertices)
        self._vao.render(mode=moderngl.POINTS, vertices=self.beam_count)

    # ---------------------------------------------------------------------
    # 4) Controls (augment Mission 2)
    # ---------------------------------------------------------------------
    def handle_key(self, key, action, modifiers, keys) -> None:
        """
        Extend Mission 2 hotkeys with Mission 3 management:
          - R : reset all beams to left margin, re-center vertically around BH
          - , : decrease vertical spacing between beams
          - . : increase vertical spacing between beams
          - (All Mission 2 keys still work: pause, size, speed, color, quit, etc.)
        """
        allowed = {getattr(keys, "ACTION_PRESS", None), getattr(keys, "ACTION_REPEAT", None)}
        if action not in allowed:
            return

        # Normalize key name
        key_name = getattr(key, "name", None) if not isinstance(key, str) else key
        up = key_name.upper() if isinstance(key_name, str) else None

        # Quit / Pause handled locally (and also by parent)
        if key == getattr(keys, "ESCAPE", None) or up == "Q":
            self.ctx._window.close()
            return
        if key == getattr(keys, "SPACE", None) or up == "SPACE":
            self.paused = not getattr(self, "paused", False)
            return

        # Reset beams to left, re-center their y with current spacing
        if up == "R":
            offsets = (np.arange(self.beam_count, dtype=np.float32)
                       - (self.beam_count - 1) * 0.5) * self.beam_spacing_px
            self._positions[:, 0] = self.left_margin_px
            self._positions[:, 1] = float(self.center[1]) + offsets
            self._vbo.write(self._positions.tobytes())
            return

        # Adjust spacing with comma/period (common choice in later missions)
        if key == getattr(keys, "COMMA", None) or up == "COMMA":
            self.beam_spacing_px = max(4.0, self.beam_spacing_px / 1.15)
            self._repack_y_positions()
            return
        if key == getattr(keys, "PERIOD", None) or up == "PERIOD":
            self.beam_spacing_px = min(80.0, self.beam_spacing_px * 1.15)
            self._repack_y_positions()
            return

        # Delegate all remaining keys (size 9/0, speed 1/2, color C, etc.) to Mission 2
        super().handle_key(key, action, modifiers, keys)

    # Utility: recompute Y around center with the current spacing
    def _repack_y_positions(self) -> None:
        offsets = (np.arange(self.beam_count, dtype=np.float32)
                   - (self.beam_count - 1) * 0.5) * self.beam_spacing_px
        self._positions[:, 1] = float(self.center[1]) + offsets
        self._vbo.write(self._positions.tobytes())

    def reset_all_beams(self) -> None:
        """Reset all beams to the left margin and re-center vertically around BH."""
        offsets = (np.arange(self.beam_count, dtype=np.float32)
                - (self.beam_count - 1) * 0.5) * self.beam_spacing_px
        self._positions[:, 0] = self.left_margin_px
        self._positions[:, 1] = float(self.center[1]) + offsets
        self._vbo.write(self._positions.tobytes())

