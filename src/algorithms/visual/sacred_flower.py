import numpy as np
from .base import VisualAlgorithm

class SacredFlowerVisual(VisualAlgorithm):
    name = "sacred_flower"

    def __init__(self, ax=None):
        self.ax = ax

    def draw(self, features: dict, memory: dict):
        rms = features.get("rms_mean", 0.05)
        centroid = features.get("centroid_mean", 1500.0)
        petals = int(6 + min(18, centroid/300))
        radius = 0.2 + min(0.7, rms*8)

        self.ax.clear()
        theta = np.linspace(0, 2*np.pi, 600)
        for k in range(petals):
            phi = 2*np.pi*k/petals
            x = radius*np.cos(theta) + 0.5*np.cos(2*theta+phi)
            y = radius*np.sin(theta) + 0.5*np.sin(2*theta+phi)
            self.ax.plot(x, y)
        self.ax.set_aspect('equal', 'box')
        self.ax.axis('off')
        self.ax.set_title(f"Sacred Flower (petals={petals})")
