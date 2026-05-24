# Activation Viz Command Runbook

This file lists the common commands for running extraction, ranking, crop/mask
rendering, and collage generation.

All examples assume:

- Repository root is the current directory.
- Dependencies are installed with `"$(which python)" -m pip install -r requirements.txt`.
- ImageNet validation data exists under `./dataset/imagenet/val`.
- The selected model is ViT-B/16 via `--model_name vit`.
- The selected channel is `652`.

## Python Command

All runnable project commands below use `"$(which python)"` inline. After you
activate `venv` or Conda, this resolves to that environment's interpreter:

```bash
which python
"$(which python)" activation_viz/run.py --help
```

Example:

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

The `$(which python)` form is bash/WSL/Conda-shell syntax. On Windows
PowerShell, activate the environment and replace `"$(which python)"` with
`python`, or use:

```powershell
(Get-Command python).Source
& (Get-Command python).Source activation_viz/run.py --help
```

If you use `activation_viz/run.py`, child scripts are launched with the same
interpreter by default. To force a specific interpreter, pass `--python_path`:

```bash
"$(which python)" activation_viz/run.py ... --python_path /path/to/python
```

Windows example:

```powershell
& (Get-Command python).Source activation_viz/run.py ... --python_path .\.venv\Scripts\python.exe
```

Conda example:

```bash
"$(which python)" activation_viz/run.py ... --python_path "$(which python)"
```

## Environment Setup

### Python `venv`

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
& (Get-Command python).Source -m pip install --upgrade pip
& (Get-Command python).Source -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
"$(which python)" -m pip install --upgrade pip
"$(which python)" -m pip install -r requirements.txt
```

### Conda

```bash
conda create -n activation-viz python=3.10 -y
conda activate activation-viz
"$(which python)" -m pip install --upgrade pip
"$(which python)" -m pip install -r requirements.txt
```

## Optional CUDA Memory Setting

If you hit CUDA out-of-memory errors caused by memory fragmentation, you can set
PyTorch's CUDA allocator config before running the pipeline. This is optional
and not needed for normal runs.

Bash:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$(which python)" activation_viz/run.py ...
```

PowerShell:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
& (Get-Command python).Source activation_viz/run.py ...
```

Use `PYTORCH_CUDA_ALLOC_CONF`, not `PYTORCH_ALLOC_CONF`.

## Change Channel Number

For `activation_viz/run.py`, change only `--channel_id`:

```bash
--channel_id 652
--channel_id 743
--channel_id 900
```

No source-code edit is required.

For direct `activation_viz/crop.py`, use `--neuron_indices`. This supports one
or more channels:

```bash
--neuron_indices 652
--neuron_indices 652 743 900
```

## One-Command Full Runs

### Whole Block 11

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

### Block 11 MLP FC2

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --step all \
  --force_layer blocks.11.mlp.fc2 \
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

### Block 11 Attention Projection

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --step all \
  --force_layer blocks.11.attn.proj \
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

## Full Run With A Different Channel

Example for channel `743`:

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --step all \
  --force_layer blocks.11.mlp.fc2 \
  --channel_id 743 \
  --pool_type raw \
  --save_values \
  --alpha_mask \
  --save_overlay \
  --results_dir results/activation_viz
```

## Fast Smoke Test

Use a small subset before running all 50k validation images:

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --step all \
  --force_layer blocks.11.mlp.fc2 \
  --channel_id 652 \
  --top_k 20 \
  --max_samples 200 \
  --pool_type raw \
  --save_values \
  --alpha_mask \
  --save_overlay \
  --results_dir results/activation_viz_smoke
```

## Step-by-Step Workflow

Run these when you want to resume, debug, or change crop parameters without
re-extracting activations.

### Step 1: Extract Activations

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --step extract \
  --force_layer blocks.11.mlp.fc2 \
  --pool_type raw \
  --save_intermediate \
  --checkpoint_interval 500 \
  --extract_batch_size 32 \
  --results_dir results/activation_viz
```

### Step 2: Rank Top Activations

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --step rank \
  --force_layer blocks.11.mlp.fc2 \
  --top_k 150 \
  --pool_type raw \
  --aggregation top_mean \
  --top_percentile 10.0 \
  --save_values \
  --results_dir results/activation_viz
```

### Step 3: Crop and Mask

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --step crop \
  --force_layer blocks.11.mlp.fc2 \
  --channel_id 652 \
  --top_k 150 \
  --pool_type raw \
  --crop_method threshold \
  --threshold_percentile 90.0 \
  --alpha_mask \
  --mask_threshold 50.0 \
  --save_values \
  --save_overlay \
  --results_dir results/activation_viz
```

### Step 4: Generate Collage

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --step collage \
  --force_layer blocks.11.mlp.fc2 \
  --channel_id 652 \
  --top_k 150 \
  --crop_method threshold \
  --results_dir results/activation_viz
```

## Direct Script Commands

These bypass `activation_viz/run.py`.

### Direct Extract

```bash
"$(which python)" activation_viz/extract.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --layers_to_hook blocks.11.mlp.fc2 output \
  --batch_size 32 \
  --pool_type raw \
  --save_intermediate \
  --checkpoint_interval 500 \
  --save_dir results/activation_viz/activations
```

### Direct Rank

```bash
"$(which python)" activation_viz/rank.py \
  --model_name vit \
  --input_file results/activation_viz/activations/vit/activations_blocks_11_mlp_fc2_output_raw.safetensors \
  --top_k 150 \
  --aggregation top_mean \
  --top_percentile 10.0 \
  --save_values \
  --output_dir results/activation_viz/top_activations
```

### Direct Crop With One Channel

```bash
"$(which python)" activation_viz/crop.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --indices_file results/activation_viz/top_activations/vit/top_activations_blocks_11_mlp_fc2_output_indices.npy \
  --values_file results/activation_viz/top_activations/vit/top_activations_blocks_11_mlp_fc2_output_values.npy \
  --activation_file results/activation_viz/activations/vit/activations_blocks_11_mlp_fc2_output_raw.safetensors \
  --neuron_indices 652 \
  --top_k_samples 150 \
  --crop_method threshold \
  --threshold_percentile 90.0 \
  --alpha_mask \
  --mask_threshold 50.0 \
  --save_overlay \
  --output_dir results/activation_viz/cropped_regions/vit/blocks_11_mlp_fc2
```

### Direct Crop With Multiple Channels

`run.py` accepts one `--channel_id` at a time. Use `crop.py` directly for
multiple channels after extraction and ranking are complete:

```bash
"$(which python)" activation_viz/crop.py \
  --model_name vit \
  --data_path ./dataset \
  --dataset imagenet-val \
  --indices_file results/activation_viz/top_activations/vit/top_activations_blocks_11_mlp_fc2_output_indices.npy \
  --values_file results/activation_viz/top_activations/vit/top_activations_blocks_11_mlp_fc2_output_values.npy \
  --activation_file results/activation_viz/activations/vit/activations_blocks_11_mlp_fc2_output_raw.safetensors \
  --neuron_indices 652 743 900 \
  --top_k_samples 150 \
  --crop_method threshold \
  --threshold_percentile 90.0 \
  --alpha_mask \
  --mask_threshold 50.0 \
  --save_overlay \
  --output_dir results/activation_viz/cropped_regions/vit/blocks_11_mlp_fc2
```

### Direct Collage

```bash
"$(which python)" tools/make_collage.py \
  --direct_dir results/activation_viz/cropped_regions/vit/blocks_11_mlp_fc2/neuron_652 \
  --channel_id 652 \
  --total_images 150 \
  --output_dir results/activation_viz/collages/Channel_652_vit_blocks_11_mlp_fc2_threshold
```

## Layer Name Reference

| Target | `--force_layer` / `--layers_to_hook` value | Filename slug |
| --- | --- | --- |
| Whole block 11 | `blocks.11` | `blocks_11` |
| Block 11 MLP FC2 | `blocks.11.mlp.fc2` | `blocks_11_mlp_fc2` |
| Block 11 attention projection | `blocks.11.attn.proj` | `blocks_11_attn_proj` |

## Output Folders For The Three ViT Targets

Whole block 11, channel 652:

```text
results/activation_viz/cropped_regions/vit/blocks_11/neuron_652/
results/activation_viz/collages/Channel_652_vit_blocks_11_threshold/
```

Block 11 MLP FC2, channel 652:

```text
results/activation_viz/cropped_regions/vit/blocks_11_mlp_fc2/neuron_652/
results/activation_viz/collages/Channel_652_vit_blocks_11_mlp_fc2_threshold/
```

Block 11 attention projection, channel 652:

```text
results/activation_viz/cropped_regions/vit/blocks_11_attn_proj/neuron_652/
results/activation_viz/collages/Channel_652_vit_blocks_11_attn_proj_threshold/
```

## Tuning Crop And Mask

Tighter crop:

```bash
--threshold_percentile 95 --mask_threshold 70
```

Looser crop:

```bash
--threshold_percentile 80 --mask_threshold 30
```

No mask, crop only:

```bash
"$(which python)" activation_viz/run.py \
  --model_name vit \
  --step crop \
  --force_layer blocks.11.mlp.fc2 \
  --channel_id 652 \
  --crop_method threshold \
  --results_dir results/activation_viz
```

Save overlay images:

```bash
--save_overlay
```

## Critical Consistency Rules

- Use `--pool_type raw` for any crop/mask visualization.
- Keep the same `--force_layer` across extract, rank, crop, and collage.
- Keep the same `--results_dir` across step-by-step commands.
- Use `--channel_id` to change the channel for `run.py`.
- Use `--neuron_indices` in `crop.py` to render multiple channels at once.
