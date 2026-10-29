from pathlib import Path

import numpy as np
import pandas as pd

from .graphs import random_graph
from .classical import brute_force_maxcut
from .qaoa import build_qaoa
from .init_strategies import random_init, linear_ramp, interp
from .optimizers import Tracker, run_optimizer

INITS = ["random", "ramp", "layerwise"]
OPTIMIZERS = ["cobyla", "adam", "spsa"]


def run_trial(g, optimal, p, init, optimizer, budget, seed, target):
    rng = np.random.default_rng(seed)  # same start for every optimizer
    m = g.number_of_edges()
    stages = range(1, p + 1) if init == "layerwise" else [p]
    tracker, x = None, None

    for q in stages:
        qnode, _, _ = build_qaoa(g, q)
        stage_budget = round(budget * q / p)  # layerwise splits budget across stages
        if tracker is None:
            tracker = Tracker(qnode, stage_budget)
        else:
            tracker.fn, tracker.budget = qnode, stage_budget

        if init == "random":
            x0 = random_init(q, rng)
        elif init == "ramp":
            x0 = linear_ramp(q)
        else:  # layerwise
            x0 = random_init(1, rng) if q == 1 else interp(x, q - 1)

        x = run_optimizer(optimizer, tracker, x0, rng)

    ratios = (m / 2 - np.array(tracker.history)) / optimal
    hit = np.where(ratios >= target)[0]
    curve = np.full(budget, ratios[-1])   # pad early-stopped runs with their final value
    k = min(len(ratios), budget)
    curve[:k] = ratios[:k]

    return {
        "final_ratio": ratios[-1],
        "evals_to_target": hit[0] + 1 if len(hit) else np.nan,
        "evals_used": tracker.count,
        "curve": curve,
    }


def main(n=8, p=2, n_graphs=30, budget=300, target=0.85):
    out = Path(__file__).resolve().parents[2] / "results"
    out.mkdir(exist_ok=True)

    rows = []
    curves={}
    for seed in range(n_graphs):
        g = random_graph(n, 0.5, seed)
        optimal, _ = brute_force_maxcut(g)
        for init in INITS:
            for opt in OPTIMIZERS:
                            r = run_trial(g, optimal, p, init, opt, budget, seed, target)
                            curves.setdefault((init, opt), []).append(r.pop("curve"))
                            rows.append({"seed": seed, "init": init, "optimizer": opt, **r})
        print(f"graph {seed + 1}/{n_graphs} done", flush=True)
        
    df = pd.DataFrame(rows)
    df.to_csv(out / "phase2_raw.csv", index=False)
    np.savez(out / "phase2_curves.npz",**{f"{i}|{o}": np.array(v) for (i, o), v in curves.items()})

    summary = (
        df.groupby(["init", "optimizer"])
        .agg(
            mean_ratio=("final_ratio", "mean"),
            median_evals_to_target=("evals_to_target", "median"),
            hit_rate=("evals_to_target", lambda s: s.notna().mean()),
        )
        .round(3)
    )
    print()
    print(summary.to_markdown())
    summary.to_csv(out / "phase2_summary.csv")

    piv = df.pivot_table(index="seed", columns=["init", "optimizer"], values="final_ratio")
    base = piv[("random", "cobyla")]
    boot_rng = np.random.default_rng(0)
    rows_g = []
    for col in piv.columns:
        d = (piv[col] - base).to_numpy()
        boots = [boot_rng.choice(d, size=len(d)).mean() for _ in range(2000)]
        rows_g.append({
            "init": col[0], "optimizer": col[1],
            "mean_gain": d.mean(),
            "ci_low": np.percentile(boots, 2.5),
            "ci_high": np.percentile(boots, 97.5),
            "win_rate": (d > 0).mean(),
        })
    gain = pd.DataFrame(rows_g).set_index(["init", "optimizer"]).round(3)
    print("\nGain over random/cobyla (paired, 95% bootstrap CI):")
    print(gain.to_markdown())
    gain.to_csv(out / "phase2_gain_vs_baseline.csv")


if __name__ == "__main__":
    main()