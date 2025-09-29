# =============================================================================
# BaseMission: Abstract base class for all simulation missions
# =============================================================================
# This is the skeleton base class for the hackathon. All missions should inherit from this.
# Implement the required methods for your own mission logic.
# =============================================================================

from abc import ABC, abstractmethod
import numpy as np
import moderngl

class BaseMission(ABC):
    """
    Base class for all simulation missions.
    """
    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        self.rs_px = min(width, height) * 0.12  # black hole radius

    @abstractmethod
    def get_name(self) -> str:
        """Return the mission name for display."""
        # TODO: Return a descriptive mission name
        pass

    @abstractmethod
    def initialize(self):
        """Initialize shaders, buffers, and mission-specific state."""
        # TODO: Set up OpenGL resources and mission state
        pass

    @abstractmethod
    def update(self, dt: float):
        """Update simulation state (called every frame)."""
        # TODO: Advance simulation by dt seconds
        pass

    @abstractmethod
    def render(self):
        """Render the mission (called every frame)."""
        # TODO: Draw the current simulation frame
        pass

    @abstractmethod
    def handle_key(self, key, action, modifiers, keys):
        """Handle keyboard input."""
        # TODO: Implement keyboard controls if needed
        pass

    def cleanup(self):
        """Optional cleanup when mission is disabled."""
        # TODO: Release resources if necessary
        pass