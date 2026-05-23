# Project Running Instructions

This file contains the complete run instructions for the neuron visualization pipeline.

The pipeline supports:

- Three XAI methods: Attention Rollout, Input x Gradient, Integrated Gradients
- Three ViT block 11 targets:
  - Full block 11: `blocks.11`
  - MLP fc2 layer: `blocks.11.mlp.fc2`
  - Attention projection layer: `blocks.11.attn.proj`
- Any target neuron/channel, for example `652` or `140`

All commands below assume Linux or Git Bash with Conda activated.

## 1. Clone Repository

```bash
git clone https://github.com/ForhanShahriarFahim/NeuronTree-AI.git
cd NeuronTree-AI/Task-12/xai_neuron_viz
```

## 2. Create Environment

```bash
conda create -n xai-neuron-viz python=3.10 -y
conda activate xai-neuron-viz
```

## 3. Install Requirements

The requirements file is:

```text
neuron_viz_pipeline/requirements.txt
```

Install all libraries:

```bash
pip install --upgrade pip
pip install -r neuron_viz_pipeline/requirements.txt
```

Main libraries used:

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

## 4. Dataset Structure

Place ImageNet validation data here:

```text
dataset/
└── imagenet/
    └── val/
        ├── n01440764/
        ├── n01443537/
        └── ...
```

The default config expects:

```text
./dataset/imagenet/val/
```

## 5. Verify Setup

```bash
$(which python) -c "import torch, timm, numpy, safetensors, yaml; print('Setup OK')"
```

Check CUDA:

```bash
$(which python) -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Check available XAI methods:

```bash
$(which python) -c "import sys; sys.path.insert(0, 'neuron_viz_pipeline'); from src.xai import available_methods; print(available_methods())"
```

Expected:

```text
['attention_rollout', 'ig', 'ixg']
```

## 6. Important CLI Arguments

### XAI Method

Choose the XAI method by changing the config file:

```text
Attention Rollout    -> neuron_viz_pipeline/configs/vit_attention_rollout.yaml
Input x Gradient     -> neuron_viz_pipeline/configs/vit_ixg.yaml
Integrated Gradients -> neuron_viz_pipeline/configs/vit_ig.yaml
```

### Target Layer

Choose the target ViT layer using `--override model.layer=...`.

```bash
--override model.layer=blocks.11
--override model.layer=blocks.11.mlp.fc2
--override model.layer=blocks.11.attn.proj
```

For all three ViT targets, keep:

```bash
--override model.layer_type=linear
```

### Target Neuron or Channel

Change both values together:

```bash
--override neuron.channel_id=<neuron_id>
--override extract.channel_id_only=<neuron_id>
```

Example for neuron `140`:

```bash
--override neuron.channel_id=140
--override extract.channel_id_only=140
```

Example for neuron `652`:

```bash
--override neuron.channel_id=652
--override extract.channel_id_only=652
```

## 7. Smoke Test

This smoke test uses only `500` images and produces `15` final results.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_attention_rollout.yaml \
  --stage all \
  --override model.layer=blocks.11.attn.proj \
  --override model.layer_type=linear \
  --override neuron.channel_id=652 \
  --override extract.channel_id_only=652 \
  --override extract.max_samples=500 \
  --override extract.batch_size=8 \
  --override extract.pool_type=raw \
  --override extract.save_intermediate=true \
  --override extract.checkpoint_interval=10 \
  --override rank.top_k=15 \
  --override rank.aggregation=top_mean \
  --override rank.top_percentile=10.0 \
  --override rank.save_values=true \
  --override crop.method=threshold \
  --override crop.threshold_percentile=90.0 \
  --override crop.alpha_mask=true \
  --override crop.mask_threshold=50.0 \
  --override crop.save_overlay=true \
  --override collage.total_images=15
```

Expected Stage 1 message:

```text
layer      : blocks.11.attn.proj
registering hook on 'blocks.11.attn.proj'
layer blocks.11.attn.proj activation shape: torch.Size([8, 197, 768])
saving only channel 652: shape torch.Size([8, 197, 1])
```

## 8. Full Dataset Runs

The commands below run on the full ImageNet validation dataset because `extract.max_samples` is not set.

Default settings:

```text
target neuron: 652
top_k: 150
batch_size: 8
ranking: top_mean, top 10 percent
crop method: threshold, 90th percentile
```

To run a different neuron, replace both:

```bash
--override neuron.channel_id=652
--override extract.channel_id_only=652
```

with:

```bash
--override neuron.channel_id=<your_neuron_id>
--override extract.channel_id_only=<your_neuron_id>
```

## 9. Attention Rollout Commands

### Attention Rollout: Full Block 11

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_attention_rollout.yaml \
  --stage all \
  --override model.layer=blocks.11 \
  --override model.layer_type=linear \
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

### Attention Rollout: Block 11 MLP fc2

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_attention_rollout.yaml \
  --stage all \
  --override model.layer=blocks.11.mlp.fc2 \
  --override model.layer_type=linear \
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

### Attention Rollout: Block 11 Attention Projection

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_attention_rollout.yaml \
  --stage all \
  --override model.layer=blocks.11.attn.proj \
  --override model.layer_type=linear \
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

## 10. Input x Gradient Commands

### Input x Gradient: Full Block 11

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_ixg.yaml \
  --stage all \
  --override model.layer=blocks.11 \
  --override model.layer_type=linear \
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

### Input x Gradient: Block 11 MLP fc2

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_ixg.yaml \
  --stage all \
  --override model.layer=blocks.11.mlp.fc2 \
  --override model.layer_type=linear \
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

### Input x Gradient: Block 11 Attention Projection

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_ixg.yaml \
  --stage all \
  --override model.layer=blocks.11.attn.proj \
  --override model.layer_type=linear \
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

## 11. Integrated Gradients Commands

Integrated Gradients is much slower than Attention Rollout and Input x Gradient because it runs many gradient steps per image.

### Integrated Gradients: Full Block 11

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_ig.yaml \
  --stage all \
  --override model.layer=blocks.11 \
  --override model.layer_type=linear \
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

### Integrated Gradients: Block 11 MLP fc2

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_ig.yaml \
  --stage all \
  --override model.layer=blocks.11.mlp.fc2 \
  --override model.layer_type=linear \
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

### Integrated Gradients: Block 11 Attention Projection

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True "$(which python)" neuron_viz_pipeline/scripts/run_stage.py \
  --config neuron_viz_pipeline/configs/vit_ig.yaml \
  --stage all \
  --override model.layer=blocks.11.attn.proj \
  --override model.layer_type=linear \
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

## 12. Output Location

Outputs are written under:

```text
neuron_viz_pipeline/results/{model}/{xai_method}/{layer_slug}/
```

Examples:

```text
neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11/
neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_mlp_fc2/
neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/
```

Inside each result folder:

```text
activations/
top_k/
xai_maps/neuron_652/
crops/neuron_652/
collages/neuron_652/
```

## 13. Clean Old Outputs

When rerunning the same method, layer, and neuron, clean old outputs first.

Example for Attention Rollout on `blocks.11.attn.proj`:

```bash
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/activations
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/top_k
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/xai_maps/neuron_652
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/crops/neuron_652
rm -rf neuron_viz_pipeline/results/vit_base_patch16_224/attention_rollout/blocks_11_attn_proj/collages/neuron_652
```

For another layer, replace the layer slug:

```text
blocks.11           -> blocks_11
blocks.11.mlp.fc2   -> blocks_11_mlp_fc2
blocks.11.attn.proj -> blocks_11_attn_proj
```

For another method, replace the method folder:

```text
attention_rollout
ixg
ig
```

For another neuron, replace:

```text
neuron_652
```

with:

```text
neuron_<id>
```

For example:

```text
neuron_140
```

## 14. Quick Reference

Change XAI method:

```bash
--config neuron_viz_pipeline/configs/vit_attention_rollout.yaml
--config neuron_viz_pipeline/configs/vit_ixg.yaml
--config neuron_viz_pipeline/configs/vit_ig.yaml
```

Change target layer:

```bash
--override model.layer=blocks.11
--override model.layer=blocks.11.mlp.fc2
--override model.layer=blocks.11.attn.proj
```

Change neuron:

```bash
--override neuron.channel_id=<id>
--override extract.channel_id_only=<id>
```

Run all stages:

```bash
--stage all
```

Use active Conda Python automatically:

```bash
"$(which python)"
```

