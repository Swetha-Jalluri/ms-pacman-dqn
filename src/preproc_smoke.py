import os, gymnasium as gym
from utils import load_config
from wrappers import PreprocessObs, FrameStack, ClipReward

# ensure ROM path is set
os.environ.setdefault("ALE_PY_ROM_DIR", os.path.expanduser("~/.ale/roms"))

cfg = load_config("configs/baseline.yaml")

env = gym.make(
    cfg["env_id"],
    full_action_space=cfg["full_action_space"],
    frameskip=cfg["frameskip"],
    repeat_action_probability=cfg["repeat_action_probability"]
)

# apply wrappers
env = PreprocessObs(env, resize=tuple(cfg["resize"]), gray=cfg["gray_scale"])
env = FrameStack(env, k=cfg["frame_stack"])
if cfg["clip_rewards"]:
    env = ClipReward(env)

obs, info = env.reset()
print("Preprocessed obs shape (C,H,W):", obs.shape)
print("Action space:", env.action_space)

# one random step
o2, r, term, trunc, _ = env.step(env.action_space.sample())
print("Step OK. Reward clip sample:", r, "Done:", term or trunc)
env.close()
