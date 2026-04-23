import os
import json
import numpy as np
import cv2
import gc
import random
from PIL import Image
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    from step2_5_utils import OdiseRefiner
except ImportError:
    print("Error: Could not import OdiseRefiner.")
    exit()

ODISE_CFG = "/root/RSVG-ZeorOV-main/ODISE/configs/Panoptic/odise_label_coco_50e.py"
ODISE_WEIGHTS = "/root/RSVG-ZeorOV-main/odise_checkpoint/odise_label_coco_50e-b67d2efc.pth"
DATA_JSON_PATH = "/root/RSVG-ZeorOV-main/data/rrsisd.json"
IMG_DIR = "/root/RSVG-ZeorOV-main/data/images"
INTERMEDIATE_DIR = "/root/RSVG-ZeorOV-main/intermediate_results" 
VIS_DIR = "/root/RSVG-ZeorOV-main/intermediate_results_vis_odise"

os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

def main():
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)["test"]
    

    samples = data[:6000] 

    print("DM")
    odise = OdiseRefiner(ODISE_CFG, ODISE_WEIGHTS)
    
    for idx, item in enumerate(tqdm(samples, desc="Processing")):
        img_id = item['iid']
        query = item['refs'][0]
        img_path = os.path.join(IMG_DIR, f"{img_id:05d}.jpg")
        
        if not os.path.exists(img_path): continue
        
        # 读取图片
        image_pil = Image.open(img_path).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w = img_cv2.shape[:2]
        

        clean_query = query.replace("-", " ").replace("_", " ").strip()
        vocab = [clean_query]
        
        candidates = []
        try:

            outputs = odise.predict_crop(img_cv2, vocab)
            if "panoptic_seg" in outputs:
                panoptic_seg, segments_info = outputs["panoptic_seg"]
                panoptic_seg = panoptic_seg.cpu().numpy()
                

                full_mask = np.zeros((h, w), dtype=np.uint8)
                for info in segments_info:
                    if info['category_id'] == 0: 
                        mask_id = info['id']
                        mask = (panoptic_seg == mask_id).astype(np.uint8)
                        if mask.shape != (h, w):
                             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        full_mask = cv2.bitwise_or(full_mask, mask)


                if full_mask.sum() > 0:
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
                    for i in range(1, num_labels):
                        candidates.append((labels == i).astype(np.bool_))
                        
        except Exception as e:
            print(f"  [!] Error processing {img_id}: {e}")
        save_name_base = f"{idx}_{img_id:05d}_odise"
        
        if candidates:
            # 1.保存 .npy 数据
            candidates_np = np.stack(candidates, axis=0)
            np.save(os.path.join(INTERMEDIATE_DIR, save_name_base + ".npy"), candidates_np)

            # --2.可视化--
            vis_img = img_cv2.copy()
            overlay = vis_img.copy()
            
            for mask in candidates:
                color = (random.randint(0, 255), random.randint(50, 255), random.randint(50, 255))
                overlay[mask] = color
                #画轮廓
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, color, 2)
                
            vis_final = cv2.addWeighted(vis_img, 0.6, overlay, 0.4, 0)
            
            cv2.putText(vis_final, f"Query: {query[:40]}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_final, f"Candidates: {len(candidates)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imwrite(os.path.join(VIS_DIR, save_name_base + "_vis.jpg"), vis_final)
            
        else:
            np.save(os.path.join(INTERMEDIATE_DIR, save_name_base + ".npy"), np.array([]))

    print(f"DM Finished")

if __name__ == "__main__":
    main()
