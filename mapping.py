import numpy as np
from skimage import color  # pip install scikit-image
import cv2

MAT_RANGE = {
  "paper":        (1.69, 5.43),
  "soft_plastic": (3.12, 7.87),
  "metal":        (8.17, 18.87),
  "hard_plastic": (9.77, 18.09),
  "glass":        (13.69, 80.7),
}
KMIN, KMAX = 1.69, 80.7

LAB_SOFT = np.array([55,  60,  40], dtype=np.float32)  # 赤寄り
LAB_HARD = np.array([80, -30, -30], dtype=np.float32)  # 水色寄り

def log_norm(k, kmin, kmax):
    k = np.clip(k, kmin, kmax)
    return (np.log(k) - np.log(kmin)) / (np.log(kmax) - np.log(kmin))

def k_to_bgr(k, kmin=1.69, kmax=80.7):
    t = float(np.clip(log_norm(k, kmin, kmax), 0.0, 1.0))
    lab = (1-t)*LAB_SOFT + t*LAB_HARD
    rgb = color.lab2rgb(lab.reshape(1,1,3)).reshape(3,)  # 0-1
    bgr = (rgb[::-1] * 255).astype(np.uint8)             # OpenCV用BGR
    return bgr

def overlay_vertical_gradient(img, x1, y1, x2, y2, bgr_top, bgr_bottom, alpha=0.35):
    """bbox内に縦グラデーションを半透明で重ねる"""
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return img

    h = y2 - y1
    w = x2 - x1

    # 0→1 の縦方向係数
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]  # (h,1)

    # (h,1,3) で色補間 → (h,w,3) に拡張
    grad = (1 - t) * bgr_top[None, None, :] + t * bgr_bottom[None, None, :]
    grad = np.repeat(grad, w, axis=1).astype(np.uint8)

    roi = img[y1:y2, x1:x2]
    blended = cv2.addWeighted(roi, 1-alpha, grad, alpha, 0)
    img[y1:y2, x1:x2] = blended
    return img

mat = label  # 例: "glass" / "metal" / "paper" / "hard_plastic" / "soft_plastic"

k_low, k_high = MAT_RANGE[mat]
bgr_top = k_to_bgr(k_low, KMIN, KMAX)
bgr_bottom = k_to_bgr(k_high, KMIN, KMAX)

overlay_vertical_gradient(result, x1, y1, x2, y2, bgr_top, bgr_bottom, alpha=0.35)

# bbox枠と文字は別で
cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,255), 2)
cv2.putText(result, f"{mat} [{k_low:.1f}-{k_high:.1f}]",
            (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
