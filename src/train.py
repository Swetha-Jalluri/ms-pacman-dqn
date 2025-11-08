# src/train.py
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from utils import load_config, set_seed, pick_device
from wrappers import PreprocessObs, FrameStack, ClipReward
from model import DQN
from replay import ReplayBuffer
from policies import epsilon_greedy, softmax_policy
from logger import CSVLogger, TrainLog

# Env vars for ALE/Gym noise reduction & ROM path
os.environ.setdefault("ALE_PY_ROM_DIR", os.path.expanduser("~/.ale/roms"))
os.environ.setdefault("GYM_DISABLE_PLUGIN_AUTOLOAD", "1")

def make_env(cfg):
    env = gym.make(
        cfg["env_id"],
        full_action_space=cfg["full_action_space"],
        frameskip=cfg["frameskip"],
        repeat_action_probability=cfg["repeat_action_probability"],
    )
    env = PreprocessObs(env, resize=tuple(cfg["resize"]), gray=cfg["gray_scale"])
    env = FrameStack(env, k=cfg["frame_stack"])
    if cfg["clip_rewards"]:
        env = ClipReward(env)
    return env

def linear_epsilon(frame, start, end, decay_frames):
    if decay_frames <= 0:
        return end
    return max(end, start - (start - end) * (frame / decay_frames))

def select_action(cfg, online, obs, device, frame):
    online.eval()
    with torch.no_grad():
        q = online(torch.from_numpy(obs[None]).to(device))
    if cfg["exploration"] == "softmax":
        eps = linear_epsilon(frame, cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay_frames"])
        temp = max(0.05, eps)
        return softmax_policy(q.squeeze(0), temp)
    else:
        eps = linear_epsilon(frame, cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay_frames"])
        return epsilon_greedy(q.squeeze(0), eps)

def train(cfg_path="configs/baseline.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    device = pick_device(cfg.get("device", "mps"))
    print("Device:", device)

    # logging
    out_dir = "runs"
    logger = CSVLogger(out_dir, cfg["run_name"])
    logger.dump_cfg(cfg)

    env = make_env(cfg)
    n_actions = env.action_space.n
    C, H, W = env.observation_space.shape

    online = DQN(C, n_actions).to(device)
    target = DQN(C, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    opt = optim.Adam(online.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.SmoothL1Loss()

    rb = ReplayBuffer(cfg["buffer_size"], obs_shape=(C, H, W))
    gamma = cfg["gamma"]

    obs, info = env.reset(seed=cfg["seed"])
    total_frames = cfg["total_frames"]
    train_freq = cfg["train_freq"]
    target_update = cfg["target_update_freq"]
    start_learn = cfg["start_learning_after"]
    batch_size = cfg["batch_size"]

    episode = 0
    ep_return, ep_len = 0.0, 0
    last_loss = None

    for frame in range(1, total_frames + 1):
        # --- act
        action = select_action(cfg, online, obs, device, frame)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rb.add(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_return += reward
        ep_len += 1

        # --- learn
        if frame > start_learn and frame % train_freq == 0 and rb.can_sample(batch_size):
            online.train()
            ob, ac, rw, nob, dn = rb.sample(batch_size)
            ob_t = torch.from_numpy(ob).to(device)
            ac_t = torch.from_numpy(ac).to(device)
            rw_t = torch.from_numpy(rw).to(device)
            nob_t = torch.from_numpy(nob).to(device)
            dn_t = torch.from_numpy(dn.astype(np.float32)).to(device)

            with torch.no_grad():
                q_next = target(nob_t).max(dim=1).values
                target_q = rw_t + gamma * (1.0 - dn_t) * q_next

            q_pred = online(ob_t).gather(1, ac_t.view(-1, 1)).squeeze(1)

            loss = loss_fn(q_pred, target_q)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            opt.step()
            last_loss = float(loss.item())

            if cfg.get("save_best", True) and logger.maybe_save_best(last_loss, online):
                torch.save(online.state_dict(), logger.model_path)

        # --- target sync
        if frame % target_update == 0:
            target.load_state_dict(online.state_dict())
            print(f"[target] updated at frame {frame}")

        # --- episode end
        if done:
            episode += 1
            logger.write(TrainLog(frame, episode, ep_return, ep_len, last_loss))
            print(f"[episode] #{episode} frame={frame} return={ep_return:.1f} len={ep_len} loss={last_loss}")
            ep_return, ep_len = 0.0, 0
            obs, info = env.reset()

    env.close()
    # always save final
    torch.save(online.state_dict(), os.path.join(logger.dir, "final.pt"))
    print("Training complete. Logs @", logger.dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = ap.parse_args()
    train(args.config)
