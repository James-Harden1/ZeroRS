"""Reference entry point for the reorganized ZeroRS inference pipeline.

Example:
python run_zerors.py --dataset-json /data/rrsisd.json --image-dir /data/images \\
  --qwen-model /models/Qwen2.5-VL-7B-Instruct --sam-checkpoint /models/sam_vit_h.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from zerors import PointBoxPromptRefiner, ZeroRSConfig, ZeroRSPipeline
from zerors.providers import OdiseCandidateProvider, QwenAttentionProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ZeroRS late-fusion referring segmentation.")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/zerors"))
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--odise-config", type=Path)
    parser.add_argument("--odise-weights", type=Path)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.odise_config) != bool(args.odise_weights):
        raise SystemExit("Provide both --odise-config and --odise-weights, or neither.")
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    config = ZeroRSConfig(
        dataset_json=args.dataset_json,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        qwen_model=args.qwen_model,
        sam_checkpoint=args.sam_checkpoint,
        odise_config=args.odise_config,
        odise_weights=args.odise_weights,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(config.qwen_model, local_files_only=True, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.qwen_model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, local_files_only=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    sam = SamPredictor(sam_model_registry["vit_h"](checkpoint=str(config.sam_checkpoint)).to(device))
    attention = QwenAttentionProvider(model, processor, config.attention_layers, config.attention_weights)
    if args.odise_config:
        from odise_refiner import OdiseRefiner  # Rename the legacy ODISE helper to this importable filename.
        diffusion = OdiseCandidateProvider(OdiseRefiner(str(args.odise_config), str(args.odise_weights)))
    else:
        diffusion = lambda _image, _query: []
    pipeline = ZeroRSPipeline(attention, diffusion, PointBoxPromptRefiner(sam, config.prompt), config)

    samples = json.loads(config.dataset_json.read_text(encoding="utf-8"))[config.split]
    if args.limit is not None:
        samples = samples[: args.limit]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples):
        image_id, query = sample["iid"], sample["refs"][0]
        image = np.asarray(Image.open(config.image_dir / f"{image_id:05d}.jpg").convert("RGB"))
        prediction = pipeline.predict(image, query)
        mask = np.zeros(image.shape[:2], dtype=bool) if prediction is None else prediction.mask
        np.save(config.output_dir / f"{index}_{image_id:05d}.npy", mask)


if __name__ == "__main__":
    main()
