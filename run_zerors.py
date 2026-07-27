"""Print the three commands that implement the original ZeroRS execution order."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the ordered ZeroRS stage commands.")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--odise-config", type=Path, required=True)
    parser.add_argument("--odise-weights", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    args = parser.parse_args()
    common = f"--dataset-json {args.dataset_json} --image-dir {args.image_dir}"
    print(f"python generate_odise_candidates.py {common} --candidate-dir {args.candidate_dir} --odise-config {args.odise_config} --odise-weights {args.odise_weights}")
    print(f"python generate_vlm_sam_candidates.py {common} --candidate-dir {args.candidate_dir} --qwen-model {args.qwen_model} --sam-checkpoint {args.sam_checkpoint}")
    print(f"python evaluate_zerors.py {common} --candidate-dir {args.candidate_dir} --output-dir {args.output_dir} --qwen-model {args.qwen_model} --sam-checkpoint {args.sam_checkpoint}" + (f" --mask-dir {args.mask_dir}" if args.mask_dir else ""))


if __name__ == "__main__":
    main()
