import os

import numpy as np

from .data import NUM_PITCHES, SEQ_LEN, roll_from_midi

NUP_BINS = np.arange(16)
ND_BINS = np.linspace(0.0, 0.15, 20)
OA_KEYS = ["nup", "ioi", "pch", "pr", "and", "nd"]


def to_active(rolls):
    arr = np.asarray(rolls)
    if arr.ndim == 4:
        return np.logical_or(arr[..., 0] > 0.5, arr[..., 1] > 0.5).astype(np.float32)
    if arr.shape[-1] == 2 * NUM_PITCHES:
        arr = arr.reshape(arr.shape[0], arr.shape[1], NUM_PITCHES, 2)
        return np.logical_or(arr[..., 0] > 0.5, arr[..., 1] > 0.5).astype(np.float32)
    return (arr > 0.5).astype(np.float32)


def load_npz(path):
    with np.load(path) as data:
        key = 'samples' if 'samples' in data else ('data' if 'data' in data else data.files[0])
        return to_active(data[key])


def load_midi_dir(path):
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(".mid"))
    return np.stack([roll_from_midi(os.path.join(path, f), SEQ_LEN) for f in files])


def normalize(hist):
    return hist / (hist.sum() + 1e-8)


def pitch_class_hist(rolls):
    counts = rolls.sum(axis=(0, 1))
    pc = np.zeros(12)
    for p in range(NUM_PITCHES):
        pc[p % 12] += counts[p]
    return normalize(pc)


def ioi_hist(rolls, max_ioi=96):
    diffs = []
    for roll in rolls:
        onset = (roll - np.vstack([np.zeros((1, roll.shape[1])), roll[:-1]])) > 0.5
        times = np.where(onset.any(axis=1))[0]
        if len(times) > 1:
            diffs.append(np.diff(times))
    hist = np.zeros(max_ioi + 1)
    if diffs:
        for d in np.concatenate(diffs):
            if d <= max_ioi:
                hist[int(d)] += 1
    return normalize(hist)


def duration_hist(rolls, max_dur=96):
    hist = np.zeros(max_dur + 1)
    for roll in rolls:
        for p in range(roll.shape[1]):
            edges = np.diff(np.concatenate([[0], roll[:, p], [0]]))
            for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                if e - s <= max_dur:
                    hist[int(e - s)] += 1
    return normalize(hist)


def pitch_range_hist(rolls):
    hist = np.zeros(NUM_PITCHES)
    for roll in rolls:
        pitches = np.where(roll > 0.5)[1]
        hist[int(pitches.max() - pitches.min()) if len(pitches) else 0] += 1
    return normalize(hist)


def bar_stats(rolls):
    peb, nupc, nup, nd, hmpe = [], [], [], [], []
    for roll in rolls:
        for b in range(8):
            rows, cols = np.where(roll[b * 16:(b + 1) * 16] > 0.5)
            pitches = np.unique(cols)
            peb.append(1.0 if len(rows) == 0 else 0.0)
            nupc.append(len(np.unique(pitches % 12)))
            nup.append(len(pitches))
            nd.append(len(rows) / (16.0 * 88.0))
            if len(cols):
                _, counts = np.unique(cols, return_counts=True)
                probs = counts / counts.sum()
                hmpe.append(-np.sum(probs * np.log2(probs + 1e-8)))
            else:
                hmpe.append(0.0)
    return {"peb": float(np.mean(peb)) * 100, "nupc": float(np.mean(nupc)), "hmpe": float(np.mean(hmpe)),
            "nup_raw": nup, "nd_raw": nd}


def histograms(rolls):
    stats = bar_stats(rolls)
    return stats, {
        "nup": normalize(np.histogram(stats["nup_raw"], bins=NUP_BINS)[0].astype(float)),
        "ioi": ioi_hist(rolls),
        "pch": pitch_class_hist(rolls),
        "pr": pitch_range_hist(rolls),
        "and": duration_hist(rolls),
        "nd": normalize(np.histogram(stats["nd_raw"], bins=ND_BINS)[0].astype(float)),
    }


def overlap_area(a, b):
    return float(np.minimum(normalize(np.asarray(a, float)), normalize(np.asarray(b, float))).sum())


def evaluate(rolls, reference_hists):
    stats, hists = histograms(rolls)
    out = {k: stats[k] for k in ("peb", "nupc", "hmpe")}
    for k in OA_KEYS:
        out[f"{k}_oa"] = overlap_area(hists[k], reference_hists[k])
    out["avg_oa"] = float(np.mean([out[f"{k}_oa"] for k in OA_KEYS]))
    return out
