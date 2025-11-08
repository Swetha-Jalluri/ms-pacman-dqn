import csv, sys, os
import matplotlib.pyplot as plt

def main(log_dir):
    csv_path = os.path.join(log_dir, "train_log.csv")
    frames, ep_returns, ep_lens, losses = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            ep_returns.append(float(row["ep_return"]))
            ep_lens.append(int(row["ep_len"]))
            val = row["loss"]
            losses.append(float(val) if val not in ("", None) else None)

    # Plot 1: Episode Return
    plt.figure()
    plt.plot(frames, ep_returns)
    plt.title("Episode Return vs Frames"); plt.xlabel("Frames"); plt.ylabel("Episode Return")
    plt.tight_layout(); plt.savefig(os.path.join(log_dir, "plot_returns.png"))

    # Plot 2: Episode Length
    plt.figure()
    plt.plot(frames, ep_lens)
    plt.title("Episode Length vs Frames"); plt.xlabel("Frames"); plt.ylabel("Episode Length")
    plt.tight_layout(); plt.savefig(os.path.join(log_dir, "plot_lengths.png"))

    # Plot 3: Loss (sparse)
    xs, ys = zip(*[(f,l) for f,l in zip(frames, losses) if l is not None]) if any(l is not None for l in losses) else ([],[])
    if xs:
        plt.figure()
        plt.plot(xs, ys)
        plt.title("Training Loss (recent samples)"); plt.xlabel("Frames"); plt.ylabel("Loss (Huber)")
        plt.tight_layout(); plt.savefig(os.path.join(log_dir, "plot_loss.png"))

    print("Saved plots to", log_dir)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
