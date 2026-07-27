"""Stage 1: generate ODISE candidate masks for the original ZeroRS pipeline."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path

from tqdm import tqdm

from zerors.data import candidate_path, load_samples, read_bgr, sample_image_path, save_masks
from zerors.odise_candidates import candidates_from_panoptic, clean_query
from zerors.odise_refiner import OdiseRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate original ODISE candidates.")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--odise-config", type=Path, required=True)
    parser.add_argument("--odise-weights", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.dataset_json, args.split, args.limit)
    odise = OdiseRefiner(str(args.odise_config), str(args.odise_weights))
    for index, sample in enumerate(tqdm(samples, desc="ODISE candidates")):
        image_id, query = sample["iid"], sample["refs"][0]
        image_path = sample_image_path(args.image_dir, image_id)
        if not image_path.exists():
            continue
        image = read_bgr(image_path)
        masks = []
        try:
            output = odise.predict_crop(image, [clean_query(query)])
            masks = candidates_from_panoptic(output, image.shape[:2])
        except Exception as error:
            print(f"ODISE failed for image {image_id}: {error}")
        save_masks(candidate_path(args.candidate_dir, index, image_id, "odise"), masks)


if __name__ == "__main__":
    main()
