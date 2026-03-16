import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
file_path = "experiment_glass_frequency_data.csv"  # 保存したCSVファイルのパス
data = pd.read_csv(file_path)

# データを確認
print(data.head())  # 最初の5行を表示

# ピーク周波数と振幅を抽出
peak_frequency = data['Frequency']
peak_amplitude = data['Amplitude']

# グラフをプロット
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], peak_frequency, label="Peak Frequency (Hz)", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Peak Frequency Over Time")
plt.legend()
plt.grid()
plt.show()

# 振幅の分布を表示
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], peak_amplitude, label="Amplitude", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Amplitude Over Time")
plt.legend()
plt.grid()
plt.show()
