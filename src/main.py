from core.loop import run_cybernetic_loop

def main():
    run_cybernetic_loop(fs=16000, chunk_duration=0.5, history=4.0)

if __name__ == "__main__":
    main()
