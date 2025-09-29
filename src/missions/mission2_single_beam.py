# =============================================================================
# Mission 2: Single Light Beam
# =============================================================================
# Purpose:
#   - Introduces dynamic simulation by animating a single light beam across the screen.
#   - Builds on Mission 1 by adding particle movement and rendering.
# Guidance for Participants:
#   - Implement a single light particle moving from left to right.
#   - Use OpenGL point rendering for the particle.
#   - Background rendering is inherited from Mission 1.
#   - Experiment with pausing and resetting the animation using keyboard controls.
# Implementation Notes:
#   - Particle position and velocity should be updated each frame.
#   - The pt_prog shader is used for particle rendering.
#   - Later missions will add more particles and interactions.
#   - (optional) Review the update and handle_key methods for animation and control logic.
# =============================================================================

import moderngl
import numpy as np
from .mission1_grid_blackhole import Mission1GridBlackHole


class Mission2SingleBeam(Mission1GridBlackHole):
    """Mission 2: Single light particle moving from left to right"""
    
    def get_name(self) -> str:
        return "Mission 2: Single Light Beam"
    
    def initialize(self):
        """Set up shaders and create initial particle"""
        self.setup_background_rendering()
        
        # Particle rendering shader
        self.pt_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            uniform float u_point_size;
            void main() {
                gl_Position = vec4(2.0 * in_pos.x / 1100.0 - 1.0, 1.0 - 2.0 * in_pos.y / 800.0, 0.0, 1.0);
                gl_PointSize = u_point_size;
            }""",
            fragment_shader="""
            #version 330
            out vec4 f_color;
            uniform vec3 u_color;
            void main() {
                f_color = vec4(u_color, 1.0);
            }"""
        )
        
        # Set particle appearance
        self.pt_prog["u_point_size"].value = 4.0
        self.pt_prog["u_color"].value = (1.0, 0.90, 0.55)  # warm yellow
        
        # Simulation state
        self.speed = 220.0  # pixels per second
        self.paused = False
        
        # Create initial particle
        self.spawn_particle()
        self.rebuild_vbo()
    
    def spawn_particle(self):
        """Create single particle at left side of screen"""
        center_y = self.height * 0.5
        x0 = -50.0  # Start off-screen to the left
        
        self.pos = np.array([[x0, center_y]], dtype=np.float32)
        self.vel = np.array([[self.speed, 0.0]], dtype=np.float32)
        self.alive = np.ones(len(self.pos), dtype=bool)
    
    def rebuild_vbo(self):
        """Update OpenGL buffer with current particle positions"""
        alive_pos = self.pos[self.alive]
        if alive_pos.size == 0:
            # Avoid zero-sized buffer issues
            alive_pos = np.zeros((1, 2), dtype=np.float32)
        self.vbo = self.ctx.buffer(alive_pos.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.pt_prog, self.vbo, "in_pos")
    
    def update(self, dt: float):
        """Update particle position and handle respawning"""
        if self.paused or len(self.pos) == 0:
            return
            
        # Move particles
        self.pos[self.alive] += self.vel[self.alive] * dt
        
        # Check if particle went off-screen to the right - respawn it
        for i in range(len(self.pos)):
            if self.alive[i] and self.pos[i, 0] > self.width + 50:
                # Respawn particle on the left
                self.pos[i, 0] = -50.0
                self.pos[i, 1] = self.height * 0.5  # Keep in center
                self.alive[i] = True
                
        # Rebuild buffer from current alive set
        self.rebuild_vbo()
    
    def render(self):
        """Render the grid background and moving particle"""
        # Render background grid and black hole from Mission 1
        self.render_background()
        
        # Draw particle points on top
        self.vao.render(mode=moderngl.POINTS)
    
    def handle_key(self, key, action, modifiers, keys):
        """Handle keyboard input for Mission 2"""
        if action == keys.ACTION_PRESS:
            if key == keys.SPACE:
                self.paused = not self.paused
                print(f"Mission 2: Animation {'paused' if self.paused else 'unpaused'}")
            elif key == keys.R:
                self.spawn_particle()
                self.rebuild_vbo()
                print("Mission 2: Particle reset")