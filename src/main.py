import os
from visualization.live_plot import live_logmel_visualizer



def main():
    live_logmel_visualizer(fs=16000, chunk_duration=0.5, history=4.0)

if __name__ == "__main__":
    main()
