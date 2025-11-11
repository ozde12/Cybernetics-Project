import numpy as np
import matplotlib.pyplot as plt
from .base import VisualAlgorithm

class LSystemVisual(VisualAlgorithm):
    name = "lsystem"

    def __init__(self, ax=None):
        self.ax = ax
        self.rules = {"F": "FF+[+F-F-F]-[-F+F+F]"}
        self.angle = 25.0
        self.iter_depth = 1

    def generate(self, axiom):
        """Apply rewrite rules iteratively."""
        s = axiom
        for _ in range(self.iter_depth):
            s = "".join(self.rules.get(ch, ch) for ch in s)
        return s

    def draw(self, features: dict, memory: dict):
        rms = features.get("rms_mean", 0.05)
        centroid = features.get("centroid_mean", 1000.0)

        # Map audio features to drawing parameters
        self.iter_depth = int(1 + min(5, centroid / 800))  # 1–5 iterations
        self.angle = 15 + min(40, rms * 4000)              # dynamic angle

        # Generate L-system string
        pattern = self.generate("F")

        # Draw
        x, y, stack = 0.0, 0.0, []
        angle = 90.0
        points_x, points_y = [x], [y]

        for ch in pattern:
            if ch == "F":
                rad = np.deg2rad(angle)
                x += np.cos(rad)
                y += np.sin(rad)
                points_x.append(x)
                points_y.append(y)
            elif ch == "+":
                angle += self.angle
            elif ch == "-":
                angle -= self.angle
            elif ch == "[":
                stack.append((x, y, angle))
            elif ch == "]" and stack:
                x, y, angle = stack.pop()
                points_x.append(None)
                points_y.append(None)

        self.ax.clear()
        self.ax.plot(points_x, points_y, color="limegreen", lw=1)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title(f"L-System (depth={self.iter_depth}, angle={self.angle:.1f})")
