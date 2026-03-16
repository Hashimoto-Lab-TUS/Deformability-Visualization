import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from datetime import datetime
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# 設定
sam_checkpoint = r"C:\Users\reooo\Desktop\dev\test_segment\sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cpu"

# 関数定義
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, edgecolor='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))  

# フォルダ内のすべての画像ファイルを取得
image_folder = r"C:\Users\reooo\Desktop\dev\test_segment\objests"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 保存先フォルダ
output_folder = r"C:\Users\reooo\Desktop\segmented_pic"

# 各画像ファイルに対して処理を実行
for image_file in image_files:
    # 画像読み込みと表示
    image = cv2.imread(image_file)
    image = cv2.resize(image, (800, 600))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.axis('on')
    #plt.show()

    # モデルの準備
    sam_model = sam_model_registry[model_type]
    sam = sam_model(checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # 対称の座標をプロンプトとして指定する
    input_point = np.array([[400, 300]])
    input_label = np.array([1])

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    #plt.show()

    # マスクの予測
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    ) 

    # マスクを適用して対象領域を抽出
    mask = masks[0]
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

    # 現在の日時を取得してファイル名に追加
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"masked_{timestamp}_{os.path.basename(image_file)}")
    cv2.imwrite(output_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

    # 対象領域を保存
    output_path = r"C:\Users\reooo\Desktop\masked_image.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

    print(f"Masked image saved to {output_path}")

    # マスクの表示
    #plt.figure(figsize=(10,10))
    #plt.imshow(image)
    #show_mask(masks[0], plt.gca()) # 最初のマスクのみ表示
    #show_points(input_point, input_label, plt.gca())
    #plt.title(f"Mask , Score: {scores[0]:.3f}", fontsize=18)
    #plt.axis('off')
    #plt.show()
