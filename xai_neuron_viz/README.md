# xai_neuron_viz

Neuron-level explainable AI visualization pipeline for studying individual neurons in CNN and Vision Transformer models.

This project extracts top-activating images for a target neuron, computes XAI saliency maps, crops the most relevant image regions, and generates collage grids for visual concept analysis.

For the full command matrix covering all XAI methods, block 11 layer variants, and neuron/channel changes, see [RUNNING_INSTRUCTIONS.md](RUNNING_INSTRUCTIONS.md).

The current focus is ViT-B/16 on ImageNet validation images, especially neuron-level visualization for layers such as:

```text
blocks.11.mlp.fc2
blocks.11.attn.proj
```

## Key Features

- Extract activations from a target model layer.
- Rank top-k images for a selected neuron/channel.
- Generate neuron-specific XAI maps.
- Crop and alpha-mask relevant image regions.
- Build collage grids for visual inspection.
- Supports three XAI methods:
  - Input x Gradient, `ixg`
  - Integrated Gradients, `ig`
  - Attention Rollout, `attention_rollout`

## Supported XAI Methods

| Method | Config | Model Support | Description |
|---|---|---|---|
| Input x Gradient | `vit_ixg.yaml`, `rn152_ixg.yaml` | ViT, ResNet | Gradient-based attribution using input multiplied by gradient |
| Integrated Gradients | `vit_ig.yaml`, `rn152_ig.yaml` | ViT, ResNet | Path-integrated gradient attribution |
| Attention Rollout | `vit_attention_rollout.yaml` | ViT only | Attention-flow based attribution using ViT attention weights |

## Project Structure

Run all commands from the project root:

```text
NeuronTree-AI/Task-12/xai_neuron_viz
```

Expected structure:

```text
xai_neuron_viz/
├── README.md
├── LICENSE
├── .gitignore
│
├── dataset/
│   └── imagenet/
│       └── val/
│           ├── n01440764/
│           ├── n01443537/
│           └── ...
│
├── data/
│   ├── __init__.py
│   └── data_proces.py
│
├── models/
│   └── __init__.py
│
├── neuron_viz_pipeline/
│   ├── requirements.txt
│   │
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── rn152_ixg.yaml
│   │   ├── rn152_ig.yaml
│   │   ├── vit_ixg.yaml
│   │   ├── vit_ig.yaml
│   │   └── vit_attention_rollout.yaml
│   │
│   ├── scripts/
│   │   ├── run_stage.py
│   │   ├── stage1_extract.py
│   │   ├── stage2_rank.py
│   │   ├── stage3_xai_maps.py
│   │   ├── stage4_crop.py
│   │   └── make_collage.py
│   │
│   ├── src/
│   │   ├── data/
│   │   ├── models/
│   │   ├── extract/
│   │   ├── rank/
│   │   ├── xai/
│   │   ├── crop/
│   │   └── utils/
│
│   └── results/                  # generated locally, ignored by git
│       └── ...
```

## Dataset Structure

The project expects ImageNet validation data in this format:

```text
dataset/
└── imagenet/
    └── val/
        ├── n01440764/
        │   ├── image_1.JPEG
        │   └── ...
        ├── n01443537/
        └── ...
```

The default config uses:

```yaml
data:
  path: "./dataset"
  dataset: "imagenet-val"
```

So the final validation directory becomes:

```text
./dataset/imagenet/val/
```

## Clone the Repository

```bash
git clone https://github.com/ForhanShahriarFahim/NeuronTree-AI.git
cd NeuronTree-AI/Task-12/xai_neuron_viz
```

## Create and Activate Environment

### Conda

```bash
conda create -n xai-neuron-viz python=3.10 -y
conda activate xai-neuron-viz
```

### Python venv

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Requirements

Run this from:

```text
NeuronTree-AI/Task-12/xai_neuron_viz
```

```bash
pip install --upgrade pip
pip install -r neuron_viz_pipeline/requirements.txt
```

Main libraries:

```text
torch
torchvision
timm
captum
numpy
scipy
pandas
safetensors
PyYAML
tqdm
Pillow
opencv-contrib-python
matplotlib
```

## Verify Installation

```bash
python -c "import torch, timm, numpy, safetensors, yaml; print('Setup OK')"
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Check available XAI methods:

```bash
python -c "import sys; sys.path.insert(0, 'neuron_viz_pipeline'); from src.xai import available_methods; print(available_methods())"
```

Expected:

```text
['attention_rollout', 'ig', 'ixg']
```

## Pipeline Stages

| Stage | Script | Description |
|---|---|---|
| Stage 1 | `stage1_extract.py` | Extract activations from target layer |
| Stage 2 | `stage2_rank.py` | Rank top-k images for each neuron |
| Stage 3 | `stage3_xai_maps.py` | Generate XAI saliency maps |
| Stage 4 | `stage4_crop.py` | Crop and alpha-mask relevant regions |
| Collage | `make_collage.py` | Create collage grids from cropped images |

All stages can be run together using:

```bash
python neuron_viz_pipeline/scripts/run_stage.py --config <config_path> --stage all
```

## Important Configuration Fields

Example ViT config:

```yaml
model:
  name: "vit"
  layer: "blocks.11.attn.proj"
  layer_type: "linear"

neuron:
  channel_id: 652
```

For ViT attention projection:

```text
blocks.11.attn.proj
```

means:

```python
model.blocks[11].attn.proj
```

This is the attention output projection layer in transformer block 11.

## Change Layer or Neuron from CLI

You can switch the target ViT layer without editing the YAML config by using `--override model.layer=...`.

Common block 11 targets:

```bash
--override model.layer=blocks.11
--override model.layer=blocks.11.mlp.fc2
--override model.layer=blocks.11.attn.proj
```

Keep the ViT layer type as linear:

```bash
--override model.layer_type=linear
```

To change the target neuron/channel, change both values:

```bash
--override neuron.channel_id=<neuron_id>
--override extract.channel_id_only=<neuron_id>
```

Example for neuron `140` at attention projection:

```bash
--override model.layer=blocks.11.attn.proj
--override model.layer_type=linear
--override neuron.channel_id=140
--override extract.channel_id_only=140
```

## Single-Neuron Extraction

For a single-neuron study, use:

```bash
--override extract.channel_id_only=<neuron_id>
```

Example:

```bash
--override neuron.channel_id=652
--override extract.channel_id_only=652
```

Both values should match.

This saves only the selected neuron during Stage 1. Without this optimization, ViT activations are saved as:

```text
[B, 197, 768]
```

With single-neuron extraction:

```text
[B, 197, 1]
```

This greatly reduces storage usage.

## Smoke Test

Run a small test before a full run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_attention_rollout.yaml \
  --stage all \
  --override neuron.channel_id=652 \
  --override extract.channel_id_only=652 \
  --override extract.max_samples=500 \
  --override extract.batch_size=8 \
  --override extract.checkpoint_interval=10 \
  --override rank.top_k=15 \
  --override collage.total_images=15
```

Expected Stage 1 log:

```text
layer      : blocks.11.attn.proj
registering hook on 'blocks.11.attn.proj'
layer blocks.11.attn.proj activation shape: torch.Size([8, 197, 768])
saving only channel 652: shape torch.Size([8, 197, 1])
```

## Full Run: Attention Rollout

The full-run commands below process the full ImageNet validation dataset because `extract.max_samples` is not set.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_attention_rollout.yaml \
  --stage all \
  --override neuron.channel_id=652 \
  --override extract.channel_id_only=652 \
  --override extract.batch_size=8 \
  --override extract.pool_type=raw \
  --override extract.save_intermediate=true \
  --override extract.checkpoint_interval=40 \
  --override rank.top_k=150 \
  --override rank.aggregation=top_mean \
  --override rank.top_percentile=10.0 \
  --override rank.save_values=true \
  --override crop.method=threshold \
  --override crop.threshold_percentile=90.0 \
  --override crop.alpha_mask=true \
  --override crop.mask_threshold=50.0 \
  --override crop.save_overlay=true \
  --override collage.total_images=150
```

## Full Run: Input x Gradient

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_ixg.yaml \
  --stage all \
  --override neuron.channel_id=652 \
  --override extract.channel_id_only=652 \
  --override extract.batch_size=8 \
  --override extract.pool_type=raw \
  --override extract.save_intermediate=true \
  --override extract.checkpoint_interval=40 \
  --override rank.top_k=150 \
  --override rank.aggregation=top_mean \
  --override rank.top_percentile=10.0 \
  --override rank.save_values=true \
  --override crop.method=threshold \
  --override crop.threshold_percentile=90.0 \
  --override crop.alpha_mask=true \
  --override crop.mask_threshold=50.0 \
  --override crop.save_overlay=true \
  --override collage.total_images=150
```

## Full Run: Integrated Gradients

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_ig.yaml \
  --stage all \
  --override neuron.channel_id=652 \
  --override extract.channel_id_only=652 \
  --override extract.batch_size=8 \
  --override extract.pool_type=raw \
  --override extract.save_intermediate=true \
  --override extract.checkpoint_interval=40 \
  --override rank.top_k=150 \
  --override rank.aggregation=top_mean \
  --override rank.top_percentile=10.0 \
  --override rank.save_values=true \
  --override crop.method=threshold \
  --override crop.threshold_percentile=90.0 \
  --override crop.alpha_mask=true \
  --override crop.mask_threshold=50.0 \
  --override crop.save_overlay=true \
  --override collage.total_images=150
```

## Run a Different Neuron

To run neuron `140`, change both values:

```bash
--override neuron.channel_id=140
--override extract.channel_id_only=140
```

Example:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_attention_rollout.yaml \
  --stage all \
  --override neuron.channel_id=140 \
  --override extract.channel_id_only=140 \
  --override extract.batch_size=8 \
  --override extract.pool_type=raw \
  --override extract.save_intermediate=true \
  --override extract.checkpoint_interval=40 \
  --override rank.top_k=150 \
  --override rank.aggregation=top_mean \
  --override rank.top_percentile=10.0 \
  --override rank.save_values=true \
  --override crop.method=threshold \
  --override crop.threshold_percentile=90.0 \
  --override crop.alpha_mask=true \
  --override crop.mask_threshold=50.0 \
  --override crop.save_overlay=true \
  --override collage.total_images=150
```

## Output Structure

Outputs are written under:

```text
neuron_viz_pipeline/results/{model}/{xai_method}/{layer_slug}/
```

Example:

```text
neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/
├── activations/
├── top_k/
├── xai_maps/
│   └── neuron_652/
├── crops/
│   └── neuron_652/
└── collages/
    └── neuron_652/
```

Important output files:

```text
xai_maps/neuron_652/attention_rollout_maps.safetensors
crops/neuron_652/rank_0000_sample_XXXX_crop.png
collages/neuron_652/Collage_Part_01.jpg
```

## Clean Old Outputs

When switching neurons or rerunning the same method/layer, clean old Stage 1/2 outputs.

For Attention Rollout:

```bash
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/activations
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/top_k
```

For Input x Gradient:

```bash
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/ixg/blocks_11_attn_proj/activations
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/ixg/blocks_11_attn_proj/top_k
```

For Integrated Gradients:

```bash
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/ig/blocks_11_attn_proj/activations
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/ig/blocks_11_attn_proj/top_k
```

## Notes

- Always run commands from the project root: `NeuronTree-AI/Task-12/xai_neuron_viz`.
- Commands use `$(which python)` inline, so the active Conda or virtual environment is detected automatically.
- For fair comparison across XAI methods, use the same target layer, neuron id, ranking settings, crop settings, and top-k.
- Attention Rollout is ViT-only.
- Integrated Gradients is much slower than Input x Gradient and Attention Rollout.
- If storage is limited, use `extract.channel_id_only`.
