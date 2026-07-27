# ZeroRS

This repository is an organized implementation of the original ZeroRS inference pipeline for zero-shot remote-sensing referring segmentation. It keeps the original candidate generation, attention verification, and fallback rules unchanged.

## Pipeline

1. `generate_odise_candidates.py` runs ODISE with the referring expression as its vocabulary, unions the category-0 panoptic regions, and saves 8-connected components.
2. `generate_vlm_sam_candidates.py` asks Qwen2.5-VL for bounding boxes, applies the original SAM box prediction plus one mask-derived box refinement, and performs box NMS.
3. `evaluate_zerors.py` extracts the Qwen attention map, verifies all saved candidates by the lexicographic score `(S_point, S_attn)`, applies the original fallback when needed, exports masks, and optionally reports RSREC/RSRES metrics.

Intermediate masks use the original names:

```text
{index}_{image_id}_odise.npy
{index}_{image_id}_sam.npy
```

## Setup

```bash
conda create -n zerors python=3.10 -y
conda activate zerors
pip install -r requirements.txt
```

Install CUDA-compatible builds of Segment Anything, ODISE, Detectron2, and `qwen-vl-utils`. Place the Qwen, SAM, and ODISE checkpoints at local paths.

## Run

Run the stages in this order. The example uses `test` and its original 6000-sample limit.

```bash
python generate_odise_candidates.py \
  --dataset-json /data/rrsisd.json \
  --image-dir /data/images \
  --candidate-dir /data/intermediate_results \
  --odise-config /models/odise_label_coco_50e.py \
  --odise-weights /models/odise_label_coco_50e-b67d2efc.pth

python generate_vlm_sam_candidates.py \
  --dataset-json /data/rrsisd.json \
  --image-dir /data/images \
  --candidate-dir /data/intermediate_results \
  --qwen-model /models/Qwen2.5-VL-7B-Instruct \
  --sam-checkpoint /models/sam_vit_h_4b8939.pth

python evaluate_zerors.py \
  --dataset-json /data/rrsisd.json \
  --image-dir /data/images \
  --candidate-dir /data/intermediate_results \
  --output-dir /data/final_best_result \
  --mask-dir /data/targetmask \
  --qwen-model /models/Qwen2.5-VL-7B-Instruct \
  --sam-checkpoint /models/sam_vit_h_4b8939.pth
```

Use `--limit N` on all three commands for a smaller consistent subset.

## Check

```bash
python -m compileall zerors generate_odise_candidates.py generate_vlm_sam_candidates.py evaluate_zerors.py
```
