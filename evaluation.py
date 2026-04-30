import os
import json
import torch
import numpy as np
import cv2
import math
import gc
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score

def mask_evaluate(pred, gt):
    assert pred.shape[-2:] == gt.shape[-2:]
    temp = pred * gt
    inter = temp.sum()
    union = ((pred + gt) - temp).sum()
    iou = inter / (union + 1e-6)
    return inter, union, iou

def box_evaluate(pred_mask, gt_box):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().detach()
    if pred_mask.dim() == 3:
        pred_mask = pred_mask[0]
    pred_mask = pred_mask.bool()

    if pred_mask.any():
        cols = pred_mask.any(dim=0)
        rows = pred_mask.any(dim=1)
        xmin = cols.nonzero(as_tuple=True)[0][0].item()
        xmax = cols.nonzero(as_tuple=True)[0][-1].item()
        ymin = rows.nonzero(as_tuple=True)[0][0].item()
        ymax = rows.nonzero(as_tuple=True)[0][-1].item()
        box_pred = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    else:
        box_pred = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0}

    box_gt = {
        "xmin": int(gt_box[0]),
        "ymin": int(gt_box[1]),
        "xmax": int(gt_box[2]),
        "ymax": int(gt_box[3]),
    }
    xA = max(box_pred["xmin"], box_gt["xmin"])
    yA = max(box_pred["ymin"], box_gt["ymin"])
    xB = min(box_pred["xmax"], box_gt["xmax"])
    yB = min(box_pred["ymax"], box_gt["ymax"])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    predW = box_pred["xmax"] - box_pred["xmin"] + 1
    predH = box_pred["ymax"] - box_pred["ymin"] + 1
    predArea = predW * predH
    gtW = box_gt["xmax"] - box_gt["xmin"] + 1
    gtH = box_gt["ymax"] - box_gt["ymin"] + 1
    gtArea = gtW * gtH
    unionArea = predArea + gtArea - interArea
    iou = interArea / (unionArea + 1e-6)
    return interArea, unionArea, iou

def Load(model_id):
    print(f"[*] Loading Qwen from {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, 
        attn_implementation="eager", device_map="cuda"
    ).eval()
    return model, processor

class SAM-Adapter:
    def __init__(self, checkpoint_path, device='cuda'):
        print(f"[*] Loading SAM from {checkpoint_path} ...")
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(device)
        self.predictor = SamPredictor(sam)
    
    def set_image(self, image_np): self.predictor.set_image(image_np)
    
    def predict_from_points(self, points):
        if not points: return None, 0.0
        input_points = np.array(points)
        input_labels = np.array([1] * len(points))
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points, point_labels=input_labels, multimask_output=True
        )
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def predict_from_box(self, box):
        # box: [x1, y1, x2, y2]
        if box is None: return None, 0.0
        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]


def region_grow(prediction_map, weighted_attn):
    
    H, W = prediction_map.shape
    visited = torch.zeros_like(prediction_map, dtype=torch.bool)
    region_mask = torch.zeros_like(prediction_map, dtype=torch.uint8)
    threshold = 0.3 # 阈值
    
    flat_attn = weighted_attn.flatten()
    unique_vals = torch.unique(flat_attn)
    unique_vals, _ = torch.sort(unique_vals, descending=True)
    top_vals = unique_vals[:7]
    
    seed_indices = []
    for val in top_vals:
        indices = (weighted_attn == val).nonzero(as_tuple=False)
        seed_indices.extend([tuple(idx.tolist()) for idx in indices])
    
    queue = []
    for y, x in seed_indices:
        queue.append((y, x))
        visited[y, x] = True
        region_mask[y, x] = 1
        
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1), (1, 0),  (1, 1)]
                  
    while queue:
        y, x = queue.pop(0)
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if not visited[ny, nx] and prediction_map[ny, nx] > threshold:
                    queue.append((ny, nx))
                    visited[ny, nx] = True
                    region_mask[ny, nx] = 1
    return region_mask


def generate_heatmap(model, processor, image_path, query):
    prompt = "The output format should be like [x1, y1, x2, y2] without any other text."
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": f"Locate it according to the following description. {query} {prompt}"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    input_ids = inputs['input_ids'][0].tolist()
    try:
        vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        pos = input_ids.index(vision_start_token_id) + 1
        vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        pos_end = input_ids.index(vision_end_token_id)
    except: return None

    image_inputs_aux = processor.image_processor(images=image_inputs)
    grid_thw = image_inputs_aux["image_grid_thw"][0].numpy()
    h_grid, w_grid = int(grid_thw[1]/2), int(grid_thw[2]/2)
    
    with torch.no_grad():
        gen_output = model.generate(**inputs, max_new_tokens=10, output_attentions=True, return_dict_in_generate=True)
        all_attentions = gen_output.attentions
    
    if all_attentions is None: return None

    num_generated = len(all_attentions)
    target_layers = [16, 17, 18, 19]
    weight_dict = {16: 0.1, 17: 0.1, 18: 0.3, 19: 0.5}
    accumulated_map = None
    for layer_idx in target_layers:
        layer_att_steps = []
        for t in range(num_generated):
            att_t = all_attentions[t][layer_idx][0, :, -1, pos:pos_end].mean(dim=0)
            layer_att_steps.append(att_t)
        att_avg_steps = torch.stack(layer_att_steps).mean(dim=0)
        try: att_2d = att_avg_steps.reshape(h_grid, w_grid).float().cpu().numpy()
        except:
            l = att_avg_steps.shape[0]; s = int(math.sqrt(l))
            att_2d = att_avg_steps[:s*s].reshape(s, s).float().cpu().numpy()
        att_2d = att_2d - att_2d.min()
        if att_2d.max() > 0: att_2d = att_2d / att_2d.max()
        w = weight_dict[layer_idx]
        if accumulated_map is None: accumulated_map = att_2d * w
        else: accumulated_map += att_2d * w
    return accumulated_map

def Triggerpoints(heatmap, fixed_threshold=0.3):
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap_u8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_blur = cv2.GaussianBlur(heatmap_u8, (7, 7), 0)
    thresh_val = int(255 * fixed_threshold)
    _, bin_map = cv2.threshold(heatmap_blur, thresh_val, 255, cv2.THRESH_BINARY)
    points = []
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bin_map, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 5: continue
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        val = heatmap_blur[cy, cx] / 255.0
        points.append((cx, cy, val))
    return points, bin_map

def condition1(mask, trigger_points):
    if mask.sum() == 0: return 9999.0
    if not trigger_points: return 9999.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_dist_global = 9999.0
    for point in trigger_points:
        px, py = point[0], point[1] 
        if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
            if mask[py, px]: return 0.0 
        dist_to_this_point = 9999.0
        for cnt in contours:
            raw_dist = cv2.pointPolygonTest(cnt, (px, py), True)
            d = abs(raw_dist) if raw_dist < 0 else 0.0
            if d < dist_to_this_point: dist_to_this_point = d
        if dist_to_this_point < min_dist_global: min_dist_global = dist_to_this_point
    return min_dist_global

def mask_score(mask, trigger_points, heatmap_full, proximity_thresh=40.0):
    score = 0.0 
    if mask.sum() == 0: return 0.0, 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for (px, py, val) in trigger_points:
        hit = False
        if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
            if mask[py, px]: hit = True
        if not hit:
            for cnt in contours:
                if cv2.pointPolygonTest(cnt, (px, py), True) >= -proximity_thresh: hit = True; break
        if hit:
            if val > 0.9: score += 1.5 
            else: score += 1.0 
    avg_heat = heatmap_full[mask].mean()
    return score, avg_heat

def max_area(heatmap, radius=20, stride=4):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
    max_x, max_y = max_loc
    threshold = max_val / 2.0
    h, w = heatmap.shape
    prompts = []
    x1 = max(0, max_x - radius); x2 = min(w, max_x + radius)
    y1 = max(0, max_y - radius); y2 = min(h, max_y + radius)
    for y in range(y1, y2, stride):
        for x in range(x1, x2, stride):
            if (x - max_x)**2 + (y - max_y)**2 <= radius**2:
                if heatmap[y, x] > threshold: prompts.append([x, y])
    if [max_x, max_y] not in prompts: prompts.append([max_x, max_y])
    return prompts

def fallback(sam_adapter, image_np, trigger_points):
    if not trigger_points: return np.zeros(image_np.shape[:2], dtype=bool), "Fallback_Failed"
    sam_adapter.set_image(image_np)
    mask, score = sam_adapter.predict_from_points(trigger_points)
    return mask, "SAM_Heatmap_Fallback"

def region_grow(heatmap_np, target_h, target_w):
    
    h_min, h_max = heatmap_np.min(), heatmap_np.max()
    heatmap_norm = (heatmap_np - h_min) / (h_max - h_min + 1e-6)
    

    heatmap_tensor = torch.from_numpy(heatmap_norm).float().cuda()
    coarse_mask_tensor = region_grow(heatmap_tensor, heatmap_tensor)
    
    coarse_mask = coarse_mask_tensor.cpu().numpy().astype(np.uint8)
    if coarse_mask.sum() == 0:
        return None

    if coarse_mask.shape != (target_h, target_w):
        coarse_mask = cv2.resize(coarse_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
    return coarse_mask > 0

def fallback2(sam_adapter, image_np, coarse_mask):
    
    if coarse_mask is None or coarse_mask.sum() == 0:
        return np.zeros(image_np.shape[:2], dtype=bool), "Fallback_Failed"
    
    ys, xs = np.where(coarse_mask)
    if len(xs) == 0 or len(ys) == 0:
         return np.zeros(image_np.shape[:2], dtype=bool), "Fallback_Failed"

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    input_box = np.array([x1, y1, x2, y2])
    
    #Box Prompt
    sam_adapter.set_image(image_np)
    mask, score = sam_adapter.predict_from_box(input_box)
    
    return mask, "SAM_RegionGrow_Box"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qwen_model, qwen_processor = Load(QWEN_PATH)
    sam_adapter = SAM-Adapter(SAM_CHECKPOINT_PATH, device=device)
    
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)["test"]

    samples = data
    print(f"Total samples to process: {len(samples)}")

    results = {'inter': [], 'union': [], 'iou': []}
    results_box = {'inter': [], 'union': [], 'iou': []}

    for idx, item in enumerate(tqdm(samples, desc="Processing")):
        img_id = item['iid']
        query = item['refs'][0]
        img_path = os.path.join(IMG_DIR, f"{img_id:05d}.jpg")
        
        if not os.path.exists(img_path): continue

        heatmap_small = generate_heatmap(qwen_model, qwen_processor, img_path, query)
        if heatmap_small is None: continue
            
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
        w_img, h_img = image_pil.size
        
        heatmap_full = cv2.resize(heatmap_small, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
        trigger_points, _ = Triggerpoints(heatmap_full, fixed_threshold=0.3)
        trigger_coords_only = [[p[0], p[1]] for p in trigger_points]
        odise_path = os.path.join(INTERMEDIATE_DIR, f"{idx}_{img_id:05d}_odise.npy")
        sam_path = os.path.join(INTERMEDIATE_DIR, f"{idx}_{img_id:05d}_sam.npy")
        
        candidates = []
        sources = [] 
        
        if os.path.exists(odise_path):
            arr = np.load(odise_path)
            if arr.size > 0:
                for i in range(arr.shape[0]): candidates.append(arr[i]); sources.append("ODISE")
        if os.path.exists(sam_path):
            arr = np.load(sam_path)
            if arr.size > 0:
                for i in range(arr.shape[0]): candidates.append(arr[i]); sources.append("SAM")

        best_mask = None
        
        if not candidates:

            coarse_mask = region_grow(heatmap_full, h_img, w_img)
            
            if coarse_mask is not None:
                best_mask, _ = fallback2(sam_adapter, image_np, coarse_mask)
            else:

                best_mask = np.zeros((h_img, w_img), dtype=bool)
        else:
            best_idx = -1
            max_hits = -1
            max_avg_heat = -1
            heatmap_norm = (heatmap_full - heatmap_full.min()) / (heatmap_full.max() - heatmap_full.min() + 1e-6)

            for i, mask in enumerate(candidates):
                if mask.shape != (h_img, w_img):
                    mask = cv2.resize(mask.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                h_hits = 0
                if mask.sum() > 0:
                     min_d = condition1(mask, trigger_points)
                     if min_d < 40: h_hits = 1 

                current_score, avg_heat = mask_score(
                    mask, trigger_points, heatmap_norm, proximity_thresh=40.0
                )
                
                if current_score > max_hits:
                    max_hits = current_score
                    max_avg_heat = avg_heat
                    best_idx = i
                elif current_score == max_hits:
                    if avg_heat > max_avg_heat:
                        max_avg_heat = avg_heat
                        best_idx = i
            
            temp_best_mask = candidates[best_idx]
            if temp_best_mask.shape != (h_img, w_img):
                 temp_best_mask = cv2.resize(temp_best_mask.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST).astype(bool)

            min_dist = condition1(temp_best_mask, trigger_coords_only)
            
            if min_dist > 100.0:
                fallback_pts = max_area(heatmap_full, radius=20)
                best_mask, _ = fallback(sam_adapter, image_np, fallback_pts)
            else:
                best_mask = temp_best_mask

        if best_mask is None: best_mask = np.zeros((h_img, w_img), dtype=bool)
        
        gt_mask_path = os.path.join(GT_MASK_DIR, f"{idx}.png")
        if not os.path.exists(gt_mask_path):
             gt_mask_path = os.path.join(GT_MASK_DIR, f"{idx:05d}.png")
        
        if os.path.exists(gt_mask_path):
            gt_mask_pil = Image.open(gt_mask_path).convert('L')
            gt_arr = np.array(gt_mask_pil)

            if gt_arr.max() > 1:
                gt_mask = torch.from_numpy(gt_arr / 255.0).float().to(device)
            else:
                gt_mask = torch.from_numpy(gt_arr).float().to(device)
            
            gt_mask = (gt_mask > 0.5).float()

            pred_mask_tensor = torch.from_numpy(best_mask).float().to(device)
            
            if pred_mask_tensor.shape != gt_mask.shape:
                pred_mask_tensor = torch.nn.functional.interpolate(
                    pred_mask_tensor[None, None, ...], 
                    size=gt_mask.shape, 
                    mode='nearest'
                )[0, 0]

            inter, union, iou = mask_evaluate(pred_mask_tensor, gt_mask)
            results['inter'].append(inter.cpu().item())
            results['union'].append(union.cpu().item())
            results['iou'].append(iou.cpu().item())

            gt_bbox = item.get('bbox', [0, 0, 0, 0])
            b_inter, b_union, b_iou = box_evaluate(pred_mask_tensor, gt_bbox)
            results_box['inter'].append(b_inter)
            results_box['union'].append(b_union)
            results_box['iou'].append(b_iou)
        else:
            results['inter'].append(0); results['union'].append(0); results['iou'].append(0)
            results_box['inter'].append(0); results_box['union'].append(0); results_box['iou'].append(0)

    #Evaluation
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    interss = results['inter']
    unionss = results['union']
    ious = results['iou']

    if len(ious) == 0:
        miou = oiou = 0.0
    else:
        miou = np.mean(ious)
        oiou = sum(interss) / (sum(unionss) + 1e-6)
        
        accs_dict = {}
        for thres in [0.3, 0.5, 0.7]:
            pred = (np.array(ious) > thres).astype(int)
            acc_val = accuracy_score(np.ones(len(ious)), pred)
            accs_dict[f'acc@{thres:.1f}'] = acc_val

        print(f'[MASK] mIoU: {miou * 100:.2f}% | oIoU: {oiou * 100:.2f}%')
        print('       ' + ' | '.join([f'{k}: {v * 100:.2f}%' for k, v in accs_dict.items()]))

    print("-" * 80)

    interss_box = results_box['inter']
    unionss_box = results_box['union']
    ious_box = results_box['iou']

    if len(ious_box) == 0:
        miou = oiou = 0.0
    else:
        miou = np.mean(ious_box)
        oiou = sum(interss_box) / (sum(unionss_box) + 1e-6)
        
        accs_dict = {}
        for thres in [0.3, 0.5, 0.7]:
            pred = (np.array(ious_box) > thres).astype(int)
            acc_val = accuracy_score(np.ones(len(ious_box)), pred)
            accs_dict[f'acc@{thres:.1f}'] = acc_val

        print(f'[BOX ] mIoU: {miou * 100:.2f}% | oIoU: {oiou * 100:.2f}%')
        print('       ' + ' | '.join([f'{k}: {v * 100:.2f}%' for k, v in accs_dict.items()]))
    
    print("=" * 80)
    print("Completed")

if __name__ == "__main__":
    main()
