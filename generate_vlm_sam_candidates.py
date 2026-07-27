"""Stage 2: generate Qwen-box SAM candidates for the original ZeroRS pipeline."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import gc
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from zerors.data import candidate_path, load_samples, read_rgb, sample_image_path, save_masks
from zerors.sam import SamAdapter
from zerors.sam_candidates import generate_sam_candidates, parse_qwen_boxes, qwen_prompt


class QwenBoxPredictor:
    def __init__(self, model_path: Path, device: str) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True
        ).eval()

    def __call__(self, image: Image.Image, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        self.model.to(self.device)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=256)
        trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        self.model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        return answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate original Qwen-box SAM candidates.")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qwen = QwenBoxPredictor(args.qwen_model, device)
    sam = SamAdapter(str(args.sam_checkpoint), device)
    samples = load_samples(args.dataset_json, args.split, args.limit)
    for index, sample in enumerate(tqdm(samples, desc="Qwen-SAM candidates")):
        image_id, query = sample["iid"], sample["refs"][0]
        image_path = sample_image_path(args.image_dir, image_id)
        if not image_path.exists():
            continue
        image_pil = Image.open(image_path).convert("RGB")
        try:
            boxes = parse_qwen_boxes(qwen(image_pil, qwen_prompt(query)), image_pil.size)
            masks = generate_sam_candidates(sam, read_rgb(image_path), boxes)
        except Exception as error:
            print(f"Qwen-SAM failed for image {image_id}: {error}")
            masks = []
        save_masks(candidate_path(args.candidate_dir, index, image_id, "sam"), masks)


if __name__ == "__main__":
    main()
