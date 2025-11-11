import numpy as np
import librosa
from .base import AudioAlgorithm

class ComplementaryHarmony(AudioAlgorithm):
    name = "complementary_harmony"

    def __init__(self, sr=16000):
        self.sr = sr # sets up the algorithm with the sample rate, the rate of the mic/audio stream runs at

    def _estimate_key(self, y):
        """
        Estimates the key of the "home pitch"

        Convert the energy into a chroma energy map, energy for each of 12 pitch classes (C, C#, D, ... B)
        Average over time to get a single vector of 12 elements
        Find the pitch class with the highest energy → root note
        Compute confidence as max energy / sum of energies

        Intuition:

        It listens for tonal bias: “Does this sound mostly like a C major world? Or maybe F-sharp?”
        Even noisy speech has harmonic partials, so the system can “guess”.
        """

        # crude: chroma → argmax; return pitch class & confidence
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        pc = chroma.mean(axis=1)
        root = int(pc.argmax())        # 0=C, 1=C#, ...
        conf = float(pc.max() / (pc.sum() + 1e-9))
        return root, conf

    def _scale_degrees(self, root):

        """
        It picks intervals +2, +5, +9 semitones above the root.

        +2 → Major 2nd

        +5 → Perfect 4th

        +9 → Major 6th (or 13th)

        These intervals are pleasant but not identical — they imply response rather than unison.
        It’s like harmonizing a voice a few steps away.
        """
        # Complement-ish set: use 2, 5, 9 semitone offsets (sus2, P5, add9 vibe)
        return [(root + i) % 12 for i in (2, 5, 9)]

    def _synth(self, freqs, dur=0.5):
        """
        Generates a time vector, t of 0.5 seconds

        For each frequency:
            Adds a sine wave + a small 2nd harmonic
        
        Then applies an exponential decay envelope
        """


        t = np.linspace(0, dur, int(self.sr*dur), endpoint=False)
        out = np.zeros_like(t)
        for f in freqs:
            out += 0.33*np.sin(2*np.pi*f*t) + 0.15*np.sin(2*np.pi*2*f*t)
        # simple percussive envelope
        env = np.exp(-5*t)
        return (out*env).astype(np.float32)

    def process(self, features: dict, memory: dict) -> dict:

        """
        Takes your latest sound chunk (recent_audio).

        Figures out its “root”.

        Builds a complementary chord around it.

        Synthesizes a quick, decaying tone.

        Sends it back to the system — which plays it out loud and stores metadata (root + confidence).
        """

        # Use recent raw audio window if available (else silence)
        y = features.get("recent_audio", np.zeros(int(self.sr*0.5), dtype=np.float32))
        root_pc, conf = self._estimate_key(y)
        degrees = self._scale_degrees(root_pc)

        # pick a register based on spectral centroid
        centroid = features.get("centroid_mean", 1500.0)
        base_freq = 110 if centroid < 1200 else 220
        freqs = [base_freq * (2 ** (d/12)) for d in [0, degrees[0], degrees[1]]]

        tone = self._synth(freqs, dur=features.get("chunk_duration", 0.5))
        return {"audio": tone, "sr": self.sr, "meta": {"root_pc": root_pc, "conf": conf}}
