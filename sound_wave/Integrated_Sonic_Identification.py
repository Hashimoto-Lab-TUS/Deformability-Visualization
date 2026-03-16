import pandas as pd
import numpy as np

# 素材のピーク周波数データ（例: 5つの素材）
materials_peak_frequencies = {
    "paper": 740,
    "soft_plastic": 388,
    "hard_plastic": 3144,
    "glass": 17158,
    "metal": 3411
}

# 素材候補のCSVを読み込む関数
def extract_valid_materials(candidate_file, valid_materials):
    # CSVを読み込み
    data = pd.read_csv(candidate_file)
    
    # "Material" 列を確認し、5つの素材と一致する行を抽出
    matched_materials = data[data["Material"].isin(valid_materials)]
    
    # 素材とそのスコアを辞書形式で返す（Scoreは数値型に変換）
    extracted_materials = {row["Material"]: float(row["Score"]) for _, row in matched_materials.iterrows()}
    return extracted_materials

# CSVデータを読み込む関数
def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# 振幅が大きい箇所のピーク周波数を抽出
def extract_peak_frequency(data, amplitude_threshold):
    # 振幅がしきい値を超える行を抽出
    significant_peaks = data[data["Amplitude"] > amplitude_threshold]

    if significant_peaks.empty:
        print("振幅がしきい値を超えるデータがありません。")
        return None

    # 振幅が最大の行を取得
    max_row = significant_peaks.loc[significant_peaks["Amplitude"].idxmax()]
    peak_frequency = max_row["Frequency"]
    return peak_frequency

# 素材を予測する関数
def predict_material(test_frequency, materials_data):
    closest_material = None
    smallest_diff = float("inf")
    
    for material, freq in materials_data.items():
        diff = abs(test_frequency - materials_peak_frequencies.get(material, 0))  # 周波数の差を計算
        if diff < smallest_diff:
            smallest_diff = diff
            closest_material = material
    
    return closest_material

# メイン処理
def main():
    # 素材候補ファイルパス（final_score.csv）
    candidate_file_path = r"C:\Users\reooo\Desktop\dev\control_dynamixel\final_scores.csv"
    
    # 素材の候補として5つの素材を指定
    valid_materials = {"paper", "soft_plastic", "hard_plastic", "glass", "metal"}

    # 素材候補を抽出
    materials_candidates = extract_valid_materials(candidate_file_path, valid_materials)

    # 素材候補が見つからない場合
    if not materials_candidates:
        print("候補となる素材が見つかりませんでした。")
        return
    
    # 入力データファイルパス
    file_path = "experiment_metal_frequency_data.csv"

    # しきい値（適切に調整してください）
    amplitude_threshold = 90000

    # CSVファイルを読み込む
    data = load_csv(file_path)

    # 振幅が大きい箇所のピーク周波数を抽出
    peak_frequency = extract_peak_frequency(data, amplitude_threshold)

    if peak_frequency is not None:
        print(f"抽出されたピーク周波数: {peak_frequency} Hz")

        # 素材候補の中から最も近いピーク周波数を持つ素材を予測
        predicted_material = predict_material(peak_frequency, materials_candidates)
        print(f"予測された素材: {predicted_material}")

if __name__ == "__main__":
    main()
