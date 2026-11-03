# BD3PM

Source code for **BD3PM: Binary Discrete Denoising Diffusion Probabilistic Model with Bidirectional Mamba for Symbolic Music Generation** (IEEE Access).

Listening examples: https://jieyuuuuu.github.io/bd3pm-demo/

## Installation

```bash
git clone https://github.com/Jieyuuuuu/BD3PM_public.git
cd BD3PM_public
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` build CUDA kernels and require Linux (or WSL2) with a CUDA toolkit matching the PyTorch build. The code was tested with Python 3.10, PyTorch 2.11 and CUDA 12.8.

## Download POP909-CL

Download the POP909-CL dataset and build the training and validation sets with the split used in the paper (`split.json`):

```bash
python prepare_data.py --midi_dir /path/to/POP909-CL/POP909_processed
```

## Training

```bash
python train.py
```

## Inference

```bash
python sample.py --ckpt ./checkpoints/bd3pm/best_ema.pt --out ./samples
```

## Evaluation

```bash
python evaluate.py --ref ./data/val.npz --samples ./samples
```
