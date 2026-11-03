python prepare_data.py --midi_dir /path/to/POP909-CL/POP909_processed
python train.py
python sample.py --ckpt ./checkpoints/bd3pm/best_ema.pt --out ./samples
python evaluate.py --ref ./data/val.npz --samples ./samples
