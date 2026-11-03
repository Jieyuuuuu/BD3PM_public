import argparse
import os

import numpy as np

from bd3pm.metrics import OA_KEYS, evaluate, histograms, load_midi_dir, load_npz

COLUMNS = ["nupc", "hmpe"] + [f"{k}_oa" for k in OA_KEYS] + ["avg_oa"]


def report(name, runs):
    mean = [np.mean([r[c] for r in runs]) for c in COLUMNS]
    std = np.std([r["avg_oa"] for r in runs], ddof=1) if len(runs) > 1 else 0.0
    print(name.ljust(10) + "".join(f"{m:9.4f}" for m in mean) + f"   +/- {std:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="data/val.npz")
    parser.add_argument("--samples", default="samples")
    args = parser.parse_args()

    ref = load_npz(args.ref)
    _, ref_hists = histograms(ref)
    print("".ljust(10) + "".join(c.rjust(9) for c in COLUMNS))

    half = len(ref) // 2
    human = evaluate(ref[:half], histograms(ref[half:])[1])
    human.update({k: v for k, v in evaluate(ref, ref_hists).items() if k in ("nupc", "hmpe")})
    report("human", [human])

    sets = sorted(d for d in os.listdir(args.samples) if d.startswith("set"))
    report("bd3pm", [evaluate(load_midi_dir(os.path.join(args.samples, d)), ref_hists) for d in sets])


if __name__ == "__main__":
    main()
