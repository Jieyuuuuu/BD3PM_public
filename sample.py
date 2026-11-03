import argparse
import math
import os

import torch

from bd3pm.data import NUM_PITCHES, SEQ_LEN, roll_to_midi
from bd3pm.diffusion import sample
from bd3pm.models import BD3PM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/bd3pm/best_ema.pt")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--sets", type=int, default=5)
    parser.add_argument("--num", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model = BD3PM().cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    model.eval()
    shape = (args.batch_size, SEQ_LEN, 2 * NUM_PITCHES)

    for seed in range(1, args.sets + 1):
        out_dir = os.path.join(args.out, f"set{seed}")
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(seed)
        k = 0
        for _ in range(math.ceil(args.num / args.batch_size)):
            for roll in sample(model, shape).cpu().numpy():
                if k < args.num:
                    roll_to_midi(roll.reshape(SEQ_LEN, NUM_PITCHES, 2), os.path.join(out_dir, f"sample_{k}.mid"))
                k += 1
        print(f"{out_dir}: {args.num} samples")


if __name__ == "__main__":
    main()
