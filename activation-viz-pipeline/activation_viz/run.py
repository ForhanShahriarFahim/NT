# =============================================================================
# Activation Viz: activation_viz/run.py  — MASTER ORCHESTRATOR
#
# NEW FILE: ties extract.py → rank.py → crop.py into one command.
# Use --step all to run everything, or --step extract/rank/crop individually.
#
# Usage:
#   # Run all three steps for ResNet-152
#   python activation_viz/run.py --model_name rn152 --step all
#
#   # Run individual steps
#   python activation_viz/run.py --model_name rn152 --step extract
#   python activation_viz/run.py --model_name rn152 --step rank
#   python activation_viz/run.py --model_name rn152 --step crop
# =============================================================================

import argparse
import os
import sys
import subprocess
import time

# Activation Viz: add project root to path so local modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# Layer and activation file defaults per model
# =============================================================================

# Activation Viz: default layer per model
LAYER_DEFAULTS = {
    'rn152':                ('layer4.2.conv3', 'output'),
    'rn50':                 ('layer4.2.conv3', 'output'),
    'vgg16':                ('features.28',    'output'),
    'vit':                  ('blocks.11',      'output'),
    'vit_base_patch16_224': ('blocks.11',      'output'),
}

# Activation Viz: expected activation file name pattern per model
def _activation_filename(model_name, layer_name, pool_type='raw'):
    safe = layer_name.replace('.', '_').replace('/', '_')
    return f"activations_{safe}_output_{pool_type}.safetensors"

def _indices_filename(layer_name):
    safe = layer_name.replace('.', '_').replace('/', '_')
    return f"top_activations_{safe}_output_indices.npy"

def _values_filename(layer_name):
    safe = layer_name.replace('.', '_').replace('/', '_')
    return f"top_activations_{safe}_output_values.npy"


def _layer_slug(layer_name):
    return layer_name.replace('.', '_').replace('/', '_')


# =============================================================================
# Step runners — call each script as subprocess so args are clean
# =============================================================================

def run_extract(args, python_path):
    """Run Step 1: extract.py."""
    layer_name, target_type = LAYER_DEFAULTS.get(
        args.model_name, ('layer4.2.conv3', 'output')
    )
    if args.force_layer:
        layer_name = args.force_layer

    print(f"\n{'='*60}")
    print(f"ACTIVATION VIZ STEP 1: Extract activations")
    print(f"  model  = {args.model_name}")
    print(f"  layer  = {layer_name}")
    print(f"  pool   = {args.pool_type}")
    print(f"{'='*60}\n")

    cmd = [
        python_path,
        os.path.join(os.path.dirname(__file__), 'extract.py'),
        '--model_name',    args.model_name,
        '--data_path',     args.data_path,
        '--dataset',       args.dataset,
        '--layers_to_hook', layer_name, target_type,
        '--batch_size',    str(args.extract_batch_size),
        '--pool_type',     args.pool_type,
        '--save_dir',      os.path.join(args.results_dir, 'activations'),
    ]
    if args.save_intermediate:
        cmd.append('--save_intermediate')
        cmd += ['--checkpoint_interval', str(args.checkpoint_interval)]
    if args.max_samples:
        cmd += ['--max_samples', str(args.max_samples)]

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"Activation Viz: extract.py failed (code {result.returncode})")
    print(f"\nActivation Viz: Step 1 done in {elapsed/60:.1f} min")
    return layer_name


def run_rank(args, python_path, layer_name):
    """Run Step 2: rank.py."""
    activation_file = os.path.join(
        args.results_dir, 'activations', args.model_name,
        _activation_filename(args.model_name, layer_name, args.pool_type)
    )

    if not os.path.exists(activation_file):
        raise FileNotFoundError(
            f"Activation Viz: Activation file not found: {activation_file}\n"
            f"Run --step extract first."
        )

    print(f"\n{'='*60}")
    print(f"ACTIVATION VIZ STEP 2: Rank top activations")
    print(f"  input  = {activation_file}")
    print(f"  top_k  = {args.top_k}")
    print(f"  aggr   = {args.aggregation}")
    print(f"{'='*60}\n")

    cmd = [
        python_path,
        os.path.join(os.path.dirname(__file__), 'rank.py'),
        '--model_name',   args.model_name,
        '--input_file',   activation_file,
        '--output_dir',   os.path.join(args.results_dir, 'top_activations'),
        '--top_k',        str(args.top_k),
        '--aggregation',  args.aggregation,
        '--top_percentile', str(args.top_percentile),
        '--batch_size',   str(args.rank_batch_size),
    ]
    if args.save_values:
        cmd.append('--save_values')

    start  = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"Activation Viz: rank.py failed (code {result.returncode})")
    print(f"\nActivation Viz: Step 2 done in {elapsed:.1f} s")
    return activation_file


def run_crop(args, python_path, layer_name, activation_file):
    """Run Step 3: crop.py."""
    indices_file = os.path.join(
        args.results_dir, 'top_activations', args.model_name,
        _indices_filename(layer_name)
    )
    values_file = os.path.join(
        args.results_dir, 'top_activations', args.model_name,
        _values_filename(layer_name)
    ) if args.save_values else None

    if not os.path.exists(indices_file):
        raise FileNotFoundError(
            f"Activation Viz: Indices file not found: {indices_file}\n"
            f"Run --step rank first."
        )

    print(f"\n{'='*60}")
    print(f"ACTIVATION VIZ STEP 3: Crop and mask images")
    print(f"  indices = {indices_file}")
    print(f"  method  = {args.crop_method}")
    print(f"  alpha   = {args.alpha_mask}")
    print(f"{'='*60}\n")

    cmd = [
        python_path,
        os.path.join(os.path.dirname(__file__), 'crop.py'),
        '--model_name',           args.model_name,
        '--data_path',            args.data_path,
        '--dataset',              args.dataset,
        '--indices_file',         indices_file,
        '--activation_file',      activation_file,
        '--output_dir',           os.path.join(args.results_dir, 'cropped_regions',
                                               args.model_name,
                                               _layer_slug(layer_name)),
        '--top_k_samples',        str(args.top_k),
        '--crop_method',          args.crop_method,
        '--threshold_percentile', str(args.threshold_percentile),
        '--crop_size',            str(args.crop_size),
        '--padding',              str(args.padding),
        '--mask_threshold',       str(args.mask_threshold),
        '--neuron_indices',       str(args.channel_id),   # channel 15 default
    ]
    if values_file:
        cmd += ['--values_file', values_file]
    if args.alpha_mask:
        cmd.append('--alpha_mask')
    if args.save_overlay:
        cmd.append('--save_overlay')

    start  = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"Activation Viz: crop.py failed (code {result.returncode})")
    print(f"\nActivation Viz: Step 3 done in {elapsed:.1f} s")


def run_collage(args, python_path, layer_name):
    """Run make_collage.py for Activation Viz output."""
    cropped_dir = os.path.join(
        args.results_dir, 'cropped_regions',
        args.model_name,
        _layer_slug(layer_name),
        f"neuron_{args.channel_id:03d}"
    )
    collage_dir = os.path.join(
        args.results_dir, 'collages',
        f"Channel_{args.channel_id}_{args.model_name}"
        f"_{_layer_slug(layer_name)}_{args.crop_method}"
    )

    if not os.path.isdir(cropped_dir):
        print(f"Activation Viz: Cropped dir not found: {cropped_dir} - skipping collage")
        return

    print(f"\n{'='*60}")
    print(f"Activation Viz: Generating collage")
    print(f"  input  = {cropped_dir}")
    print(f"  output = {collage_dir}")
    print(f"{'='*60}\n")

    # Use make_collage directly with --direct_dir.
    cmd = [
        python_path,
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'make_collage.py'),
        '--direct_dir',  cropped_dir,
        '--channel_id',  str(args.channel_id),
        '--total_images', str(args.top_k),
        '--output_dir',  collage_dir,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Activation Viz: make_collage.py returned non-zero - "
              f"check output above")


# =============================================================================
# Activation Viz CLI args
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Activation Viz - Master runner (extract -> rank -> crop)"
    )

    parser.add_argument('--model_name', type=str, default='rn152',
                        choices=['rn152', 'rn50', 'vgg16', 'vit',
                                 'vit_base_patch16_224'])
    parser.add_argument('--data_path',  type=str, default='./dataset')
    parser.add_argument('--dataset',    type=str, default='imagenet-val')
    parser.add_argument('--force_layer', type=str, default=None,
                        help="Override default layer for model")
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'extract', 'rank', 'crop', 'collage'],
                        help="Which step to run")
    parser.add_argument('--results_dir', type=str,
                        default='results/activation_viz',
                        help="Root directory for all Activation Viz outputs")
    parser.add_argument('--python_path', type=str,
                        default=sys.executable,
                        help="Python interpreter path")

    # Extract args
    parser.add_argument('--pool_type', type=str, default='raw',
                        choices=['raw', 'gap', 'gmp'])
    parser.add_argument('--extract_batch_size', type=int, default=32)
    parser.add_argument('--save_intermediate', action='store_true')
    parser.add_argument('--checkpoint_interval', type=int, default=500)
    parser.add_argument('--max_samples', type=int, default=None)

    # Rank args
    parser.add_argument('--top_k',          type=int,   default=150)
    parser.add_argument('--aggregation',    type=str,   default='top_mean',
                        choices=['max', 'mean', 'sum', 'top_mean'])
    parser.add_argument('--top_percentile', type=float, default=10.0)
    parser.add_argument('--rank_batch_size', type=int,  default=1000)
    parser.add_argument('--save_values',    action='store_true')

    # Crop args
    parser.add_argument('--crop_method',           type=str,   default='threshold',
                        choices=['threshold', 'bbox', 'center'])
    parser.add_argument('--threshold_percentile',  type=float, default=90.0)
    parser.add_argument('--crop_size',             type=int,   default=224)
    parser.add_argument('--padding',               type=int,   default=20)
    parser.add_argument('--alpha_mask',            action='store_true')
    parser.add_argument('--mask_threshold',        type=float, default=50.0)
    parser.add_argument('--save_overlay',          action='store_true')
    parser.add_argument('--channel_id',            type=int,   default=15)

    return parser.parse_args()


def main():
    args = get_args()

    # expand ~ in python path
    python_path = os.path.expanduser(args.python_path)

    print(f"\nActivation Viz [run]:")
    print(f"  model       = {args.model_name}")
    print(f"  step        = {args.step}")
    print(f"  results_dir = {args.results_dir}")
    print(f"  channel_id  = {args.channel_id}")
    print()

    os.makedirs(args.results_dir, exist_ok=True)

    # resolve layer
    layer_name, _ = LAYER_DEFAULTS.get(args.model_name, ('layer4.2.conv3', 'output'))
    if args.force_layer:
        layer_name = args.force_layer

    total_start = time.time()

    if args.step in ('all', 'extract'):
        layer_name = run_extract(args, python_path)

    activation_file = os.path.join(
        args.results_dir, 'activations', args.model_name,
        _activation_filename(args.model_name, layer_name, args.pool_type)
    )

    if args.step in ('all', 'rank'):
        activation_file = run_rank(args, python_path, layer_name)

    if args.step in ('all', 'crop'):
        run_crop(args, python_path, layer_name, activation_file)

    if args.step in ('all', 'collage'):
        run_collage(args, python_path, layer_name)

    total_elapsed = time.time() - total_start
    print(f"\nActivation Viz: All steps complete in {total_elapsed/60:.1f} min")
    print(f"Results -> {args.results_dir}/")


if __name__ == "__main__":
    main()
