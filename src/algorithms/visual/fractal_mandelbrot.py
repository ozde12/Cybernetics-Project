import numpy as np
import matplotlib.pyplot as plt
from .base import VisualAlgorithm

class MandelbrotVisual(VisualAlgorithm):
    name = "fractal_mandelbrot"

    def __init__(self, ax=None):
        self.ax = ax
        self.x0, self.y0, self.zoom = -0.5, 0.0, 1.5
        self.vmin, self.vmax = 0, 255  # for stable colors

    def draw(self, features: dict, memory: dict):
        rms = features.get("rms_mean", 0.05)
        centroid = features.get("centroid_mean", 1500.0)

        # update zoom from sound, but CLAMP it
        factor = 1 - 0.2 * np.clip(rms * 10, 0, 0.9)  # in (0.82 .. 1]
        self.zoom *= factor
        self.zoom = float(np.clip(self.zoom, 0.2, 10.0))  # <-- clamp to avoid 1/0

        # iterations scale with "brightness"
        iters = int(50 + min(300, centroid / 8))

        # build finite grid (avoid inf)
        w, h = 400, 300
        span_x = 1.5 / self.zoom
        span_y = 1.125 / self.zoom
        x = np.linspace(self.x0 - span_x, self.x0 + span_x, w)
        y = np.linspace(self.y0 - span_y, self.y0 + span_y, h)
        C = x + 1j * y[:, None]

        Z = np.zeros_like(C, dtype=np.complex128)
        M = np.zeros(C.shape, dtype=np.int32)

        # avoid runtime warnings for overflow/invalid
        with np.errstate(over='ignore', invalid='ignore'):
            for i in range(iters):
                Z = Z * Z + C
                diverged = (np.abs(Z) > 2) & (M == 0)
                M[diverged] = i
                Z[diverged] = 0

        # replace any NaNs/Infs in the mask, just in case
        if not np.isfinite(M).all():
            M = np.nan_to_num(M, nan=iters, posinf=iters, neginf=0)

        self.ax.clear()
        self.ax.imshow(M, origin='lower', aspect='auto', cmap='magma',
                       vmin=self.vmin, vmax=max(self.vmin+1, M.max()))
        self.ax.set_title(f"Mandelbrot (zoom={self.zoom:.2f}, iters={iters})")
        self.ax.axis('off')
