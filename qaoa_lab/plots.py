from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"random": "tab:gray", "ramp": "tab:red", "layerwise": "tab:blue"}
STYLES = {"cobyla": "-", "adam": "--", "spsa": ":"}


def main():
    out = Path(__file__).resolve().parents[2] / "results"
    data = np.load(out / "phase2_curves.npz")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key in data.files:
        init, opt = key.split("|")
        arr = data[key]                      # shape: (n_graphs, budget)
        x = np.arange(1, arr.shape[1] + 1)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ax.plot(x, mean, color=COLORS[init], linestyle=STYLES[opt],
                label=f"{init} / {opt}", linewidth=1.8)
        if init == "ramp" and opt == "cobyla" or init == "random" and opt == "cobyla":
            ax.fill_between(x, mean - sem, mean + sem, color=COLORS[init], alpha=0.15)

    ax.axhline(0.85, color="black", linewidth=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.99, 0.852, "target 0.85", ha="right", fontsize=8)
    ax.set_xlabel("Circuit evaluations")
    ax.set_ylabel("Best-so-far approximation ratio (mean over graphs)")
    ax.set_title("QAOA p=2 on 8-node Max-Cut: convergence by init and optimizer")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "phase2_convergence.png", dpi=200)
    print("saved", out / "phase2_convergence.png")


if __name__ == "__main__":
    main()