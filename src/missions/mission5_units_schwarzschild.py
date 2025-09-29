# =============================================================================
# Mission 5: SI Units & Schwarzschild Radius
# =============================================================================
# Purpose:
#   - Introduces real-world physics by using SI units (metres, seconds) and the Schwarzschild radius.
#   - Demonstrates how to convert between simulation pixels and physical units.
# Guidance for Participants:
#   - Set up the world using SI units and appropriate conversions for rendering.
#   - Calculate the black hole radius using the Schwarzschild formula: r_s = 2GM/c^2.
#   - No light bending yet; beams move in straight lines.
#   - (optional if time left) Experiment with different black hole masses and observe the effect on the simulation.
# Implementation Notes:
#   - Use pixels_per_metre for unit conversion.
#   - All positions, velocities, and sizes should be in SI units.
#   - Later missions will introduce relativistic effects and light bending.
#   - Review the initialize and spawn_multiple_beams methods for unit conversion logic.
# =============================================================================

import numpy as np
import moderngl
from .mission4_multiple_beams import Mission4MultipleBeams

class Mission5UnitsSchwarzschild(Mission4MultipleBeams):
    """
    Mission 5: Multiple parallel light beams, now using SI units and Schwarzschild radius.
    - All positions, velocities, and sizes are in SI units (metres, seconds, etc.).
    - The black hole's radius is set by the Schwarzschild formula: r_s = 2GM/c^2.
    - No light bending yet; beams move in straight lines.
    - Participants must set up the world using SI units and appropriate conversions.
    """
    # Physical constants (SI units)
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    c = 299_792_458  # m/s

    def __init__(self, ctx, w, h):
        super().__init__(ctx, w, h)
        self.window_width = w
        self.window_height = h

    def get_name(self) -> str:
        return "Mission 5: SI Units & Schwarzschild Radius"

    def initialize(self):
        """Set up SI unit world and black hole parameters, and prepare for polar coordinates"""
        # --- SI world setup ---
        # Set pixels_per_metre so the Schwarzschild radius is about 40 pixels
        self.pixels_per_metre = 0.05  # 1 metre = 0.05 pixels, so 40 pixels = 800 metres in this mapping
        self.bh_mass_kg = 2.0e30      # roughly a solar mass

        # Compute Schwarzschild radius in metres
        self.rs_m = 2 * self.G * self.bh_mass_kg / (self.c ** 2)
        self.rs_px = self.rs_m * self.pixels_per_metre

        # Set world size in metres to match window size in pixels
        self.world_width_m = self.window_width / self.pixels_per_metre
        self.world_height_m = self.window_height / self.pixels_per_metre

        # Centre of black hole in world coordinates (pixels)
        # Use float32 to avoid integer truncation when sending to GPU
        self.center = np.array([self.window_width / 2.0, self.window_height / 2.0], dtype=np.float32)

        # Set up beam speed (speed of light)
        self.speed_m_s = self.c
        self.speed = self.speed_m_s * self.pixels_per_metre  # pixels/sec

        # Rendering dimensions (in pixels); should match window dimensions
        self.width = int(self.window_width)
        self.height = int(self.window_height)

        # Spacing in metres (vertical distance between beams)
        self.spacing_m = self.world_height_m / 20.0  # 20 beams vertically
        self.spacing = max(2, int(self.spacing_m * self.pixels_per_metre))

        # Prepare polar coordinates array (will be set in spawn_multiple_beams)
        self.polar = None  # Will hold (r, theta) for each particle

        # Internal placeholders for black hole rendering
        self._bh_disc_prog = None
        self._bh_disc_vbo = None
        self._bh_disc_vao = None

        # Call parent to finish setup (will call spawn_particle / spawn_multiple_beams)
        super().initialize()

    def spawn_multiple_beams(self):
        margin_m = self.world_height_m / 20.0
        margin_px = int(margin_m * self.pixels_per_metre)
        ys = np.arange(margin_px, self.height - margin_px + 1, self.spacing, dtype=np.int32)
        count_per_beam = 80
        x0_m = -self.world_width_m * 0.1  # Start just off the left edge (in metres)
        x0 = x0_m * self.pixels_per_metre  # convert to pixels

        pts = []
        vels = []
        # Use a consistent pixel step for particles along the beam (e.g., 1.5 pixels)
        step_px = 1.5
        for y in ys:
            for i in range(count_per_beam):
                pts.append([x0 - i * step_px, float(y)])
                vels.append([self.speed, 0.0])

        self.pos = np.array(pts, dtype=np.float32)
        self.vel = np.array(vels, dtype=np.float32)
        self.alive = np.ones(len(self.pos), dtype=bool)

        # Compute initial polar coordinates (r, theta) relative to black hole centre
        to_center = self.pos - self.center
        r = np.sqrt(to_center[:, 0] ** 2 + to_center[:, 1] ** 2)
        theta = np.arctan2(to_center[:, 1], to_center[:, 0])
        self.polar = np.stack([r, theta], axis=1)

    def update(self, dt: float):
        """Update multiple particles with collision detection (no light bending), and update polar coordinates"""
        if self.paused or len(self.pos) == 0:
            return
        self.pos[self.alive] += self.vel[self.alive] * dt
        to_center = self.pos - self.center
        r = np.sqrt(to_center[:, 0] ** 2 + to_center[:, 1] ** 2)
        theta = np.arctan2(to_center[:, 1], to_center[:, 0])
        self.polar = np.stack([r, theta], axis=1)

        # Use polar radius for collision detection (should match rendered black hole)
        hit = self.polar[:, 0] <= self.rs_px
        self.alive &= ~hit

        # Off-screen check (right); keep tolerant margin
        offscreen_right = self.pos[:, 0] > (self.width + 50)
        self.alive &= ~offscreen_right

        if not np.any(self.alive):
            self.spawn_multiple_beams()
        self.rebuild_vbo()

    def _ensure_bh_disc(self):
        """Create the program, VBO and VAO for the black hole disc if not already created."""
        if self._bh_disc_prog is not None:
            return

        self._bh_disc_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;         // unit circle coords or centre (in pixels-space before scaling)
                uniform vec2 u_center;   // pixel coords of center
                uniform float u_radius;  // radius in pixels
                uniform vec2 u_view;     // (window_width, window_height) in pixels
                void main() {
                    // in_vert is used as unit circle coordinates for ring vertices,
                    // and (0,0) for the center vertex.
                    vec2 pixel = in_vert * u_radius + u_center;
                    // Convert pixels -> NDC:
                    // x_ndc = (pixel.x / width) * 2 - 1
                    // y_ndc = 1 - (pixel.y / height) * 2   (flip Y because pixel y grows downwards)
                    vec2 ndc = (pixel / u_view) * 2.0 - 1.0;
                    ndc.y = -ndc.y; // flip Y to convert top-left pixel coords -> OpenGL NDC
                    gl_Position = vec4(ndc, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 f_color;
                void main() {
                    f_color = vec4(0.0, 0.0, 0.0, 1.0); // Black disc
                }
            '''
        )

        # Build TRIANGLE_FAN with centre as first vertex, then ring vertices, closing the ring by repeating the first ring vertex
        n = 64
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ring = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype('f4')
        # centre + ring + first_ring to close
        verts = np.vstack((np.array([0.0, 0.0], dtype='f4'), ring, ring[0:1].astype('f4'))).astype('f4')
        self._bh_disc_vbo = self.ctx.buffer(verts.tobytes())
        self._bh_disc_vao = self.ctx.simple_vertex_array(
            self._bh_disc_prog, self._bh_disc_vbo, 'in_vert'
        )

    def render(self):
        """Render the beams (via parent) and then the black hole as a disc in pixel coordinates"""
        # First, render the parent (grid, beams, etc.)
        # Use getattr to safely call parent's render if present
        parent_render = getattr(super(), 'render', None)
        if parent_render is not None:
            parent_render()

        # Now ensure black hole disc resources exist and render it
        self._ensure_bh_disc()

        # Set uniforms in pixel coordinates
        # u_view must be floats
        self._bh_disc_prog['u_view'].value = (float(self.window_width), float(self.window_height))
        # Ensure center is provided as two floats
        self._bh_disc_prog['u_center'].value = (float(self.center[0]), float(self.center[1]))
        self._bh_disc_prog['u_radius'].value = float(self.rs_px)

        # Draw the filled disc
        self._bh_disc_vao.render(moderngl.TRIANGLE_FAN)

