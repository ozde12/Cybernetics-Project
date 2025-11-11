import numpy as np

def choose_audio_algorithm(features: dict):
    rms = features["rms_mean"]        # scalar
    zcr = features["zcr_mean"]
    tonal = features.get("tonal_conf", 0.0)  # from chroma/key estimation
    tempo = features.get("tempo", 0)

    # Simple rule hierarchy (start here; evolve later)
    if tonal > 0.6 and 0.02 < rms < 0.2:
        return "complementary_harmony"
    if zcr > 0.15 or rms >= 0.2:
        return "granular_echo"
    if tempo > 60:
        return "rhythm_responder"
    # default fallback
    return "granular_echo"

def choose_visual_algorithm(features: dict):
    centroid = features["centroid_mean"]
    onset_rate = features.get("onset_rate", 0.0)
    tonal = features.get("tonal_conf", 0.0)

    if tonal > 0.6:
        return "sacred_flower"
    if onset_rate > 2.0:
        return "lsystem"
    return "fractal_mandelbrot"
