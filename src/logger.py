import csv, os, json, time
from dataclasses import dataclass, asdict

@dataclass
class TrainLog:
    frame: int
    episode: int
    ep_return: float
    ep_len: int
    loss: float | None

class CSVLogger:
    def __init__(self, out_dir: str, run_name: str):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(out_dir, f"{run_name}-{ts}")
        os.makedirs(self.dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, "train_log.csv")
        self.cfg_path = os.path.join(self.dir, "config.json")
        self.model_path = os.path.join(self.dir, "best.pt")
        self._loss_best = None

        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame","episode","ep_return","ep_len","loss"])

    def dump_cfg(self, cfg: dict):
        with open(self.cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def write(self, rec: TrainLog):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([rec.frame, rec.episode, rec.ep_return, rec.ep_len, rec.loss if rec.loss is not None else ""])

    def maybe_save_best(self, loss_val, model):
        if loss_val is None: return False
        if self._loss_best is None or loss_val < self._loss_best:
            self._loss_best = loss_val
            return True
        return False
