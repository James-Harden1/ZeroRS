"""Stage 3: attention verification, fallback, prediction export, and evaluation."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from zerors.attention import extract_attention_map, extract_trigger_points, resize_attention
from zerors.data import candidate_path, ground_truth_path, load_masks, load_samples, read_rgb, sample_image_path
from zerors.metrics import box_iou, mask_box, mask_iou, summarize
from zerors.sam import SamAdapter
from zerors.selection import select_or_fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run original ZeroRS candidate verification and evaluation.")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=6000)
    return parser.parse_args()


def load_attention_model(model_path: Path, device: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=device,
    ).eval()
    return model, processor


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_attention_model(args.qwen_model, device)
    sam = SamAdapter(str(args.sam_checkpoint), device)
    mask_ious: list[float] = []
    mask_intersections: list[int] = []
    mask_unions: list[int] = []
    box_ious: list[float] = []
    box_intersections: list[int] = []
    box_unions: list[int] = []

    for index, sample in enumerate(tqdm(load_samples(args.dataset_json, args.split, args.limit), desc="ZeroRS evaluation")):
        image_id, query = sample["iid"], sample["refs"][0]
        image_path = sample_image_path(args.image_dir, image_id)
        if not image_path.exists():
            continue
        attention_grid = extract_attention_map(model, processor, str(image_path), query)
        if attention_grid is None:
            continue
        image = read_rgb(image_path)
        attention = resize_attention(attention_grid, image.shape[:2])
        triggers, _ = extract_trigger_points(attention)
        candidates = [
            *load_masks(candidate_path(args.candidate_dir, index, image_id, "odise"), image.shape[:2]),
            *load_masks(candidate_path(args.candidate_dir, index, image_id, "sam"), image.shape[:2]),
        ]
        result = select_or_fallback(image, attention, candidates, triggers, sam)
        mask = np.asarray(result.mask, dtype=bool)
        np.save(args.output_dir / f"{index}_{image_id:05d}_mask.npy", mask)
        np.save(args.output_dir / f"{index}_{image_id:05d}_attention.npy", attention)

        if args.mask_dir is None:
            continue
        target_path = ground_truth_path(args.mask_dir, index)
        if target_path is None:
            continue
        target = np.asarray(Image.open(target_path).convert("L")) > 0
        if mask.shape != target.shape:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((target.shape[1], target.shape[0]), Image.Resampling.NEAREST), dtype=bool)
        iou, intersection, union = mask_iou(mask, target)
        box_score, box_intersection, box_union = box_iou(mask_box(mask), sample.get("bbox", [0, 0, 0, 0]))
        mask_ious.append(iou)
        mask_intersections.append(intersection)
        mask_unions.append(union)
        box_ious.append(box_score)
        box_intersections.append(box_intersection)
        box_unions.append(box_union)

    if args.mask_dir is not None:
        for label, values in (("RSRES", summarize(mask_ious, mask_intersections, mask_unions)), ("RSREC", summarize(box_ious, box_intersections, box_unions))):
            print(label)
            print(" | ".join(f"{key}: {value * 100:.2f}%" for key, value in values.items()))


if __name__ == "__main__":
    main()
