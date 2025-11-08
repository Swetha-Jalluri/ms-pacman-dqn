import os, gymnasium as gym

# Ensure ALE can find ROMs
os.environ.setdefault("ALE_PY_ROM_DIR", os.path.expanduser("~/.ale/roms"))

def main():
    env = gym.make("ALE/MsPacman-v5")  # relies on shimmy registering ALE
    obs, info = env.reset()
    print("Env OK:", env.action_space, getattr(obs, "shape", None))
    # Take 10 random steps
    total_r = 0.0
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        total_r += r
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print("Random run reward (10 steps):", total_r)

if __name__ == "__main__":
    main()
