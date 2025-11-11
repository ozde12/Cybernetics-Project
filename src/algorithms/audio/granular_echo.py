import numpy as np
import random
from .base import AudioAlgorithm

"""
Keep a rolling memory (~2 seconds) of the most recent mic audio.

Pick many small “grains” (~50 ms) from random places in that memory.

Jitter each grain: slight random time-stretch / pitch shift (e.g., 0.8–1.2×) via interpolation.

Window the grain with a Hann envelope (soft in/out).

Overlap-add all grains at random start positions into the current output buffer (same length as your chunk).

Normalize to keep it safe, return the texture.

"""


class GranularEcho(AudioAlgorithm):
    name = "granular_echo"

    def __init__(self, sr=16000):
        self.sr = sr
        self.memory = np.zeros(int(sr * 5), dtype=np.float32)  # ~5 s rolling buffer

    def process(self, features: dict, memory: dict) -> dict:
        y = features.get("recent_audio", np.zeros(int(self.sr * 0.5), dtype=np.float32))
        chunk_dur = features.get("chunk_duration", 0.5)
        hop = int(0.05 * self.sr)  # ~50 ms grains

        # update memory buffer
        self.memory = np.concatenate([self.memory[len(y):], y])

        # choose grain positions from recent memory
        grains = []
        for _ in range(20):  # 20 grains per output
            start = random.randint(0, len(self.memory) - hop - 1)
            grain = self.memory[start:start + hop].copy()
            # apply random pitch shift
            rate = random.uniform(0.8, 1.2)
            grain = np.interp(np.arange(0, len(grain), rate),
                              np.arange(len(grain)), grain)
            # envelope
            env = np.hanning(len(grain))
            grains.append(grain * env)

        # overlap-add the grains
        out_len = int(chunk_dur * self.sr)
        output = np.zeros(out_len, dtype=np.float32)
        for g in grains:
            start = random.randint(0, max(1, out_len - len(g)))
            end = start + len(g)
            output[start:end] += g[:out_len - start]

        # normalize
        if np.max(np.abs(output)) > 0:
            output /= np.max(np.abs(output))

        return {"audio": output.astype(np.float32), "sr": self.sr, "meta": {"grains": len(grains)}}
