import sys
import numpy as np
import os, json, cv2, random
import torch
from PIL import Image
import torch.nn.functional as F

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from pathlib import Path
ROOT = Path(__file__).resolve().parent        
CENTERNET_ROOT = Path("/home/reo/dev/Multimodal/CenterNet2") 

# 最優先でパスを通す
sys.path.insert(0, str(ROOT))           # Detic のルート
sys.path.insert(0, str(CENTERNET_ROOT)) # CenterNet2 のルート

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from centernet.config import add_centernet_config
from detic.config import add_detic_config

import detic, centernet
print("detic:", detic.__file__)
print("centernet:", centernet.__file__)

# 素材分類モデル（ResNet）
sys.path.insert(0, "/home/reo/dev/Multimodal/pytorch-material-classification/experiments/gtos_mobile.finetune.resnet") 
from segmented_objects_inference import (
    load_model as load_mat_model,
    infer_material_from_pil,
    class_labels as MATERIAL_LABELS,
)

mat_model, mat_device = load_mat_model()
print("material model device:", mat_device)

# Detic 設定
cfg_path = ROOT / "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(str(cfg_path))
print("has DETIC:", hasattr(cfg.MODEL, "DETIC"))

cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
predictor = DefaultPredictor(cfg)

# 硬さマッピング（素材分布→色）
# 予備実験データから素材ごとの min/max を採用
MAT_RANGE = {
    "glass":   (13.69, 20.00),
    "metal":   (8.17,  18.87),
    "paper":   (1.69,  5.43),
    "plastic": (3.12,  7.87),  # soft/hard を統合（現状のモデル出力が plastic なので）
    # "wood":  (?, ?)  
}

# 全体レンジ（全測定値 min/max）
KMIN, KMAX = 1.69, 20.00

# Lab補間の端点（赤→水色）
LAB_SOFT = np.array([55,  60,  40], dtype=np.float32)  # 柔らかい(危険)=赤
LAB_HARD = np.array([80, -30, -30], dtype=np.float32)  # 硬い(安全)=水色

# lab2rgb 変換用
try:
    from skimage import color as skcolor
except ImportError:
    raise ImportError("scikit-image が必要です: pip install scikit-image")

def log_norm(k, kmin, kmax):
    k = float(np.clip(k, kmin, kmax))
    return (np.log(k) - np.log(kmin)) / (np.log(kmax) - np.log(kmin))

def k_to_bgr(k, kmin=KMIN, kmax=KMAX):
    t = float(np.clip(log_norm(k, kmin, kmax), 0.0, 1.0))
    lab = (1 - t) * LAB_SOFT + t * LAB_HARD
    rgb = skcolor.lab2rgb(lab.reshape(1, 1, 3)).reshape(3,)  # 0..1
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr = (rgb[::-1] * 255).astype(np.uint8)
    return bgr

def overlay_vertical_gradient(img, x1, y1, x2, y2, bgr_top, bgr_bottom, alpha=0.35):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # clip
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)

    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return

    # 3ch BGRで扱う
    if img.ndim == 2:
        img[:] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] != 3:
        return

    # grad生成（uint8, (h,w,3)）
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]         # (h,1)
    top = np.array(bgr_top, dtype=np.float32).reshape(1,1,3)
    bot = np.array(bgr_bottom, dtype=np.float32).reshape(1,1,3)
    grad = (1.0 - t) * top + t * bot                                 # (h,1,3)
    grad = np.repeat(grad, w, axis=1).astype(np.uint8)               # (h,w,3)

    roi = img[y1:y2, x1:x2]
    if roi.shape[:2] != grad.shape[:2] or roi.shape[2] != 3:
        return

    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, grad, alpha, 0.0)


def get_material_range(label: str):
    """モデル出力ラベル→(k_low,k_high) を返す。未定義なら None"""
    lab = label.lower().strip()
    return MAT_RANGE.get(lab, None)

def overlay_masked_vertical_gradient(img, mask, x1, y1, x2, y2, bgr_top, bgr_bottom, alpha=0.75):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return

    # bbox領域に対する縦グラデーション
    patch = make_vertical_grad(h, w, bgr_top, bgr_bottom)

    roi_img = img[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    if roi_mask.shape[:2] != roi_img.shape[:2]:
        return

    blended = cv2.addWeighted(roi_img, 1.0 - alpha, patch, alpha, 0.0)

    # mask部分だけ反映
    roi_img[roi_mask] = blended[roi_mask]
    img[y1:y2, x1:x2] = roi_img

# 画像読込
im = cv2.imread("/home/reo/Downloads/IMG_4617.jpg")
if im is None:
    raise FileNotFoundError("画像が読み込めませんでした ><")

# 物体検出（Detic）
outputs = predictor(im)
instances = outputs["instances"].to("cpu")

print("has masks:", instances.has("pred_masks")) 

boxes = instances.pred_boxes.tensor.numpy().astype(int) if len(instances) > 0 else []
mat_labels, mat_scores = [], []

scores = instances.scores.numpy() if len(instances) > 0 else []
masks  = instances.pred_masks.numpy() if len(instances) > 0 else []

# 小さい/大きいBBoxフィルタ
H, W = im.shape[:2]
IMG_AREA = H * W

MIN_SIDE = 40
MIN_AREA = 80 * 80
MIN_SCORE = 0.6

MAX_SIDE_RATIO = 0.9
MAX_AREA_RATIO = 0.6

keep_indices = []

for i, ((x1, y1, x2, y2), sc) in enumerate(zip(boxes, scores)):
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h

    if w < MIN_SIDE or h < MIN_SIDE:
        continue
    if area < MIN_AREA:
        continue
    if sc < MIN_SCORE:
        continue
    if w > W * MAX_SIDE_RATIO or h > H * MAX_SIDE_RATIO:
        continue
    if area > IMG_AREA * MAX_AREA_RATIO:
        continue

    keep_indices.append(i)

boxes = boxes[keep_indices]
scores = scores[keep_indices]
masks = masks[keep_indices]

# ROI→素材推定（ResNet）
mat_labels, mat_scores = [], []

for (x1, y1, x2, y2) in boxes:
    x1, y1 = max(0, x1), max(0, y1)
    crop = im[y1:y2, x1:x2]
    if crop.size == 0:
        mat_labels.append("n/a")
        mat_scores.append(0.0)
        continue

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)

    label, probs = infer_material_from_pil(mat_model, pil, mat_device, verbose=False)
    score = float(np.max(probs))

    mat_labels.append(label)
    mat_scores.append(score)

print("num instances after filter:", len(boxes))
print("material labels:", mat_labels)
print("material scores:", [round(s, 3) for s in mat_scores])

# 可視化用関数
def paste_patch_with_mask(overlay_img, overlay_a, patch, mask, x1, y1, x2, y2, alpha=0.9):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(overlay_img.shape[1], x2)
    y2 = min(overlay_img.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return

    roi_img = overlay_img[y1:y2, x1:x2]
    roi_a   = overlay_a[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    if roi_mask.shape[:2] != roi_img.shape[:2]:
        return

    roi_img[roi_mask] = patch[roi_mask]
    roi_a[roi_mask] = np.maximum(roi_a[roi_mask], alpha)

    overlay_img[y1:y2, x1:x2] = roi_img
    overlay_a[y1:y2, x1:x2] = roi_a

def make_vertical_grad(h, w, bgr_top, bgr_bottom):
    t = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
    top = np.array(bgr_top, dtype=np.float32).reshape(1, 1, 3)
    bot = np.array(bgr_bottom, dtype=np.float32).reshape(1, 1, 3)
    grad = (1.0 - t) * top + t * bot
    grad = np.repeat(grad, w, axis=1)
    return grad.astype(np.uint8)

# 硬さグラデーション生成
overlay_img = np.zeros_like(im, dtype=np.uint8)
overlay_a   = np.zeros((im.shape[0], im.shape[1]), np.float32)

ALPHA = 0.90

for i, (x1, y1, x2, y2) in enumerate(boxes):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(im.shape[1], x2)
    y2 = min(im.shape[0], y2)

    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        continue

    label = mat_labels[i]
    rng = get_material_range(label)
    if rng is None:
        continue

    k_low, k_high = rng
    bgr_top = k_to_bgr(k_low)
    bgr_bot = k_to_bgr(k_high)

    patch = make_vertical_grad(h, w, bgr_top, bgr_bot)

    paste_patch_with_mask(
        overlay_img,
        overlay_a,
        patch,
        masks[i],
        x1, y1, x2, y2,
        alpha=ALPHA
    )

# 元画像と合成
a3 = overlay_a[..., None]
result = (im.astype(np.float32) * (1.0 - a3) + overlay_img.astype(np.float32) * a3).astype(np.uint8)

# bboxと文字
for i, (x1, y1, x2, y2) in enumerate(boxes):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(result, (x1, y1), (x2, y2), (0,255,255), 2)

    txt = f"{mat_labels[i]} {mat_scores[i]*100:.0f}%"
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 2.0
    THICKNESS = 3
    cv2.putText(result, txt, (x1, max(0, y1-10)),
                FONT, FONT_SCALE, (0,255,255), THICKNESS, cv2.LINE_AA)
    
def resize_to_fit(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # 拡大はしない
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

show = resize_to_fit(result, max_w=1600, max_h=900)
cv2.imshow("result", show)
cv2.waitKey(0)
cv2.destroyAllWindows()