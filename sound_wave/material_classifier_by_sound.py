import pandas as pd
import numpy as np

# 素材のピーク周波数データを辞書として定義
materials_peak_frequencies = {
    "paper": 740,
    "soft_plastic": 388,
    "hard_plastic": 3144,
    "glass": 17158,
    "metal": 3411
}

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
        diff = abs(test_frequency - freq)  # 周波数の差を計算
        if diff < smallest_diff:
            smallest_diff = diff
            closest_material = material
    
    return closest_material

# メイン処理
def main():
    # 入力ファイルパス
    file_path = "test_paper_frequency_data.csv"

    # しきい値（）
    amplitude_threshold = 90000

    # CSVファイルを読み込む
    data = load_csv(file_path)

    # 振幅が大きい箇所のピーク周波数を抽出
    peak_frequency = extract_peak_frequency(data, amplitude_threshold)

    if peak_frequency is not None:
        print(f"抽出されたピーク周波数: {peak_frequency} Hz")

        # 素材を予測
        predicted_material = predict_material(peak_frequency, materials_peak_frequencies)
        print(f"予測された素材: {predicted_material}")

if __name__ == "__main__":
    main()
