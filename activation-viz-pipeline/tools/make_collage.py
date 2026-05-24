import argparse
import os
from pathlib import Path

from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(
        description="Create rank-grid collages from Activation Viz crop output."
    )
    parser.add_argument(
        "--direct_dir",
        "--input_dir",
        dest="input_dir",
        required=True,
        help="Directory containing rank_XXXX_sample_XXXXX_crop images.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where collage JPEGs will be written.",
    )
    parser.add_argument(
        "--channel_id",
        type=int,
        required=True,
        help="Channel / neuron index shown in collage titles.",
    )
    parser.add_argument(
        "--total_images",
        type=int,
        default=150,
        help="Number of ranked images to include.",
    )
    parser.add_argument(
        "--images_per_grid",
        type=int,
        default=15,
        help="Images per collage grid. Default 15 gives 3x5 grids.",
    )
    return parser.parse_args()


def _rank_key(path: Path) -> int:
    try:
        return int(path.name.split("_")[1])
    except (IndexError, ValueError):
        return 10**9


def _find_crop_images(input_dir: Path):
    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.name.startswith("rank_")
            and ("_crop.png" in path.name or "_crop.jpg" in path.name)
            and "_no_mask" not in path.name
            and "_overlay" not in path.name
            and "_info" not in path.name
        ],
        key=_rank_key,
    )


def create_collages(
    input_dir: str,
    channel_id: int,
    total_images: int,
    images_per_grid: int,
    output_dir: str,
):
    import matplotlib.pyplot as plt

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    crop_images = _find_crop_images(input_path)
    if not crop_images:
        raise FileNotFoundError(
            f"No Activation Viz crop images found in: {input_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    selected = crop_images[:total_images]
    while len(selected) < total_images:
        selected.append(None)

    rows, cols = 3, 5
    num_collages = (total_images + images_per_grid - 1) // images_per_grid

    print("Activation Viz [make_collage]:")
    print(f"  input_dir  = {input_path}")
    print(f"  output_dir = {output_path}")
    print(f"  images     = {min(len(crop_images), total_images)}")

    for collage_idx in range(num_collages):
        fig, axes = plt.subplots(rows, cols, figsize=(15, 9))

        start_rank = collage_idx * images_per_grid
        end_rank = min(start_rank + images_per_grid, total_images) - 1
        fig.suptitle(
            f"Channel {channel_id} | Ranks {start_rank + 1}-{end_rank + 1}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        for i, ax in enumerate(axes.flatten()):
            global_rank = start_rank + i
            image_path = selected[global_rank] if global_rank < len(selected) else None

            if image_path is not None and image_path.exists():
                ax.imshow(Image.open(image_path).convert("RGB"))
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No image\n{global_rank + 1}",
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=9,
                )

            ax.set_title(f"Rank {global_rank + 1}", fontsize=11)
            ax.axis("off")

        plt.subplots_adjust(wspace=0, hspace=0.15)
        save_path = output_path / f"Collage_Part_{collage_idx + 1:02d}.jpg"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved = {save_path}")


def main():
    args = get_args()
    create_collages(
        input_dir=args.input_dir,
        channel_id=args.channel_id,
        total_images=args.total_images,
        images_per_grid=args.images_per_grid,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
