import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import pick_device
from wrappers import PreprocessObs, FrameStack, ClipReward
from model import DQN


def make_env(cfg, rgb_array=False):
    """Create eval env; if rgb_array=True, enable frame capture."""
    env_kwargs = dict(
        full_action_space=cfg.get("full_action_space", False),
        frameskip=cfg.get("frameskip", 4),
        repeat_action_probability=cfg.get("repeat_action_probability", 0.25),
    )
    if rgb_array:
        env_kwargs["render_mode"] = "rgb_array"

    env = gym.make(cfg["env_id"], **env_kwargs)
    env = PreprocessObs(env,
                        resize=tuple(cfg.get("resize", [84, 84])),
                        gray=cfg.get("gray_scale", True))
    env = FrameStack(env, k=cfg.get("frame_stack", 4))
    if cfg.get("clip_rewards", True):
        env = ClipReward(env)
    return env


def load_cfg(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.json found in {run_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def load_model(run_dir: str, cfg, device):
    C, H, W = cfg.get("obs_shape", (4, 84, 84))  # fallback; real shape from env later
    n_actions = cfg.get("n_actions", 9)          # fallback; real count from env later

    # We'll rebuild properly once we see env spaces; this is just placeholder
    net = DQN(C, n_actions).to(device)

    best_path = os.path.join(run_dir, "best.pt")
    final_path = os.path.join(run_dir, "final.pt")

    ckpt_path = None
    if os.path.exists(best_path):
        ckpt_path = best_path
    elif os.path.exists(final_path):
        ckpt_path = final_path

    if ckpt_path is None:
        raise FileNotFoundError(f"No model found at {best_path} or {final_path}")

    state = torch.load(ckpt_path, map_location=device)
    try:
        net.load_state_dict(state)
    except RuntimeError:
        # If shape mismatch because of placeholder, caller will rebuild after env init.
        raise

    return net, ckpt_path


def greedy_action(net, obs, device):
    net.eval()
    with torch.no_grad():
        q = net(torch.from_numpy(obs[None]).to(device))
        return int(q.argmax(dim=1).item())


def save_frame(frame, path):
    """Save an RGB numpy frame as PNG."""
    # frame: (H, W, 3) uint8
    plt.imsave(path, frame)


def run_eval(run_dir, episodes=10, save_json=True, save_frames=False):
    device = pick_device("mps")
    print("Device:", device)

    cfg = load_cfg(run_dir)
    rgb_mode = bool(save_frames)

    # Build env with/without rgb_array
    env = make_env(cfg, rgb_array=rgb_mode)

    # Now that env exists, get correct shapes for model
    n_actions = env.action_space.n
    C, H, W = env.observation_space.shape

    net = DQN(C, n_actions).to(device)

    best_path = os.path.join(run_dir, "best.pt")
    final_path = os.path.join(run_dir, "final.pt")
    if os.path.exists(best_path):
        ckpt_path = best_path
    elif os.path.exists(final_path):
        ckpt_path = final_path
    else:
        raise FileNotFoundError(f"No model found at {best_path} or {final_path}")

    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    net.eval()

    print(f"Evaluating model: {ckpt_path}")

    returns = []
    lengths = []

    # choose some steps in ep1 to snapshot
    snapshot_steps = {50, 150, 300, 600}
    frame_id = 0

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            action = greedy_action(net, obs, device)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_return += reward
            steps += 1

            if save_frames and ep == 1 and steps in snapshot_steps:
                # Only works if env was created with render_mode="rgb_array"
                frame = env.render()
                if frame is not None:
                    out_path = os.path.join(run_dir, f"frame_{steps}.png")
                    save_frame(frame, out_path)
                    print(f"[frame] saved {out_path}")
                    frame_id += 1

        returns.append(ep_return)
        lengths.append(steps)
        print(f"[eval] ep={ep}/{episodes} return={ep_return:.1f} steps={steps}")

    env.close()

    avg_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    avg_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))

    print("\n=== Evaluation Summary ===")
    print(f"Model: {ckpt_path}")
    print(f"Episodes: {episodes}")
    print(f"Avg Return: {avg_ret:.2f}  (± {std_ret:.2f})")
    print(f"Avg Steps : {avg_len:.1f}  (± {std_len:.1f})")

    if save_json:
        summary = {
            "model": os.path.basename(ckpt_path),
            "episodes": episodes,
            "returns": returns,
            "avg_return": avg_ret,
            "std_return": std_ret,
            "avg_steps": avg_len,
            "std_steps": std_len,
        }
        out_path = os.path.join(run_dir, "eval_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("Saved eval_summary.json to", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="Run directory containing config.json and model checkpoints")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--save_frames", action="store_true",
                    help="Save a few rgb_array frames as PNGs for the report")
    args = ap.parse_args()

    run_eval(args.run_dir,
             episodes=args.episodes,
             save_json=args.save_json,
             save_frames=args.save_frames)
