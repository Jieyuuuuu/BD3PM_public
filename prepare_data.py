import argparse
import json
import os

import numpy as np

from bd3pm.data import song_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", required=True)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--split", default="split.json")
    args = parser.parse_args()

    split = json.load(open(args.split))
    songs = {song for part in split.values() for song, _ in part}
    windows = {s: song_windows(os.path.join(args.midi_dir, f"{s}.mid")) for s in sorted(songs)}
    os.makedirs(args.out_dir, exist_ok=True)
    for name, items in split.items():
        arr = np.stack([windows[song][i] for song, i in items])
        np.savez_compressed(os.path.join(args.out_dir, f"{name}.npz"), samples=arr)
        print(name, arr.shape)


if __name__ == "__main__":
    main()
