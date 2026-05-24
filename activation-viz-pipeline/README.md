# Activation Viz

Activation Viz is a lightweight pipeline for finding and visualizing the
ImageNet validation images that most strongly activate individual model
channels. It supports CNN and ViT backbones, stores activation tensors once,
ranks samples per channel, then renders activation-guided crops and masks.

The current project is activation-based only. It does not run CRP, IG, IxG, or
attention rollout.

## Features

- Extract raw activations from a selected model layer.
- Rank top-k dataset samples for every channel.
- Crop and mask images using the selected channel's spatial activation map.
- Generate rank-grid collages for quick inspection.
- Run the whole workflow from CLI without editing source files.
- Select ViT layer target and channel number from CLI.

## Supported Models

| CLI name | Model |
| --- | --- |
| `rn50` | torchvision ResNet-50 |
| `rn152` | torchvision ResNet-152 |
| `vgg16` | torchvision VGG-16 |
| `vit` | timm `vit_base_patch16_224` |
| `vit_base_patch16_224` | timm `vit_base_patch16_224` |

## Repository Structure

```text
CoE-preprocessing/
├── activation_viz/
│   ├── __init__.py
│   ├── run.py              # Orchestrates extract -> rank -> crop -> collage
│   ├── extract.py          # Forward hooks and activation tensor storage
│   ├── rank.py             # Chunked top-k ranking per channel
│   └── crop.py             # Activation-map crop, mask, overlay rendering
├── data/
│   ├── __init__.py
│   ├── data_proces.py      # ImageNet ImageFolder dataset wrapper
│   └── download_imagenet.py
├── dataset/
│   └── imagenet/
│       └── val/
│           ├── n01440764/
│           │   └── *.JPEG
│           └── ...
├── models/
│   └── __init__.py         # Model factory
├── tools/
│   └── make_collage.py     # Collage utility
├── COMMANDS.md             # Full command runbook
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

`dataset/` is intentionally gitignored because ImageNet and generated artifacts
should not be committed.

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd CoE-preprocessing
```

### Option A: Python `venv`

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install requirements from the activated environment in bash/WSL/Conda shell:

```bash
"$(which python)" -m pip install --upgrade pip
"$(which python)" -m pip install -r requirements.txt
```

PowerShell equivalent:

```powershell
& (Get-Command python).Source -m pip install --upgrade pip
& (Get-Command python).Source -m pip install -r requirements.txt
```

### Option B: Conda

Create and activate a Conda environment:

```bash
conda create -n activation-viz python=3.10 -y
conda activate activation-viz
```

Install requirements in bash/WSL/Conda shell:

```bash
"$(which python)" -m pip install --upgrade pip
"$(which python)" -m pip install -r requirements.txt
```

PowerShell equivalent:

```powershell
& (Get-Command python).Source -m pip install --upgrade pip
& (Get-Command python).Source -m pip install -r requirements.txt
```

After activating `venv` or Conda, the commands below use `"$(which python)"`
inline so the active environment interpreter is used directly.

### Locate The Active Python

On macOS/Linux/WSL, or Conda shells on Linux servers, you can resolve the active
environment interpreter dynamically:

```bash
which python
"$(which python)" activation_viz/run.py --help
```

Run project commands with `"$(which python)"` at the start:

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --step all \
  --force_layer blocks.11.mlp.fc2 \
  --channel_id 652 \
  --pool_type raw \
  --save_values \
  --alpha_mask \
  --save_overlay \
  --results_dir results/activation_viz
```

On Windows PowerShell:

```powershell
(Get-Command python).Source
& (Get-Command python).Source activation_viz/run.py --help
```

If CUDA memory fragmentation causes out-of-memory errors, optionally set
PyTorch's CUDA allocator configuration before the command. This is not required
for normal runs:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$(which python)" activation_viz/run.py ...
```

PowerShell equivalent:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
& (Get-Command python).Source activation_viz/run.py ...
```

## Dataset Layout

The dataset loader expects ImageNet validation data in `torchvision.datasets.ImageFolder`
format:

```text
dataset/
└── imagenet/
    └── val/
        ├── n01440764/
        │   ├── ILSVRC2012_val_00000293.JPEG
        │   └── ...
        ├── n01443537/
        └── ...
```

Use `--data_path ./dataset` when running commands. The code appends
`imagenet/val` internally.

## Output Layout

By default, outputs are written under `results/activation_viz/`:

```text
results/activation_viz/
├── activations/{model_name}/
│   └── activations_{layer}_output_raw.safetensors
├── top_activations/{model_name}/
│   ├── top_activations_{layer}_output_indices.npy
│   ├── top_activations_{layer}_output_values.npy
│   └── top_activations_{layer}_output_metadata.json
├── cropped_regions/{model_name}/{layer_slug}/neuron_{channel_id}/
│   ├── rank_0000_sample_{idx}_crop.png
│   ├── rank_0000_sample_{idx}_crop_no_mask.jpg
│   ├── rank_0000_sample_{idx}_crop_info.json
│   └── rank_0000_sample_{idx}_overlay.jpg
└── collages/
    └── Channel_{channel_id}_{model_name}_{layer_slug}_{crop_method}/
        └── Collage_Part_01.jpg
```

## ViT Layers

You can choose all requested ViT targets from CLI with `--force_layer`:

| Target | CLI value |
| --- | --- |
| Whole block 11 output | `blocks.11` |
| Block 11 MLP output projection | `blocks.11.mlp.fc2` |
| Block 11 attention output projection | `blocks.11.attn.proj` |

You can choose the channel from CLI with `--channel_id`. You do not need to
edit source files to change channel number.

Examples:

```bash
--channel_id 652
--channel_id 743
--channel_id 900
```

When using `activation_viz/run.py`, change the channel only in `--channel_id`.
When using `activation_viz/crop.py` directly, use `--neuron_indices`; that
direct command also supports multiple channels at once.

## Quick Start

Run the full workflow for ViT block 11, channel 652:

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --step all \
  --force_layer blocks.11 \
  --channel_id 652 \
  --top_k 150 \
  --pool_type raw \
  --aggregation top_mean \
  --top_percentile 10.0 \
  --crop_method threshold \
  --threshold_percentile 90.0 \
  --alpha_mask \
  --mask_threshold 50.0 \
  --save_values \
  --save_overlay \
  --save_intermediate \
  --checkpoint_interval 500 \
  --extract_batch_size 32 \
  --results_dir results/activation_viz
```

For all layer/channel command variants, see [COMMANDS.md](COMMANDS.md).

## Important Notes

- Use `--pool_type raw` for crop and mask rendering. `gap` and `gmp` remove
  spatial information and will not produce activation masks.
- For ViT sequence outputs shaped `[N, 197, 768]`, ranking ignores the CLS token
  and uses only patch tokens. Crop/mask rendering also removes the CLS token and
  reshapes the remaining 196 tokens to a `14 x 14` grid.
- If you run `--step extract`, `--step rank`, and `--step crop` separately, keep
  `--force_layer`, `--pool_type`, and `--results_dir` consistent across steps.

## Troubleshooting

If extraction uses too much memory:

```bash
--save_intermediate --checkpoint_interval 500
```

If crop images are all black or too tight:

```bash
--mask_threshold 30
```

If crops are too loose:

```bash
--threshold_percentile 95 --mask_threshold 70
```

If a layer name is not found, inspect the timm model module names:

```python
import timm
model = timm.create_model("vit_base_patch16_224", pretrained=True)
for name, _ in model.named_modules():
    print(name)
```
