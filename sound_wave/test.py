import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# マイクの設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # サンプリング周波数 44100 Hz
CHUNK = 1024  # バッファサイズ

# ファイルに保存する設定
name = input("Enter material name: ")
output_file = name + "_frequency_decay.csv"

# PyAudioオブジェクトを作成
p = pyaudio.PyAudio()

# ストリームを開始
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("マイク入力を開始...")

# 記録する周波数と許容誤差
target_frequencies = [8000, 3000]  # トラッキングしたい周波数
freq_tolerance = 1000  # 許容誤差 ±500 Hz

# グラフの設定
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

time_data = []  # 時間データ
amplitude_data_3000 = []  # 3000 Hz帯域の振幅
amplitude_data_5000 = []  # 5000 Hz帯域の振幅

line_3000, = ax.plot([], [], label=f"{target_frequencies[0]} Hz ± {freq_tolerance} Hz", color="blue")
line_5000, = ax.plot([], [], label=f"{target_frequencies[1]} Hz ± {freq_tolerance} Hz", color="red")

ax.set_xlim(0, 2)  # 時間軸（10秒分を表示）
ax.set_ylim(0, 5000)  # 振幅の範囲
ax.set_title("Amplitude Decay of Specific Frequencies")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()

# 周波数データの保存準備
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Frequency (Hz)", "Amplitude"])  # ヘッダー行を追加

    start_time = time.time()

    try:
        while True:
            # 音声データを取得
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

            # FFTを計算
            fft_data = np.abs(np.fft.fft(data))[:CHUNK // 2]
            freqs = np.fft.fftfreq(CHUNK, 1 / RATE)[:CHUNK // 2]

            # 各周波数帯域の振幅を計算
            amplitudes = []
            for target_freq in target_frequencies:
                indices = np.where((freqs >= target_freq - freq_tolerance) &
                                   (freqs <= target_freq + freq_tolerance))[0]
                amplitude = np.sum(fft_data[indices]) if len(indices) > 0 else 0
                amplitudes.append(amplitude)

            # 時間を記録
            elapsed_time = time.time() - start_time

            # データを保存
            for freq, amp in zip(target_frequencies, amplitudes):
                writer.writerow([elapsed_time, freq, amp])

            # データをグラフ用に更新
            time_data.append(elapsed_time)
            amplitude_data_3000.append(amplitudes[0])
            amplitude_data_5000.append(amplitudes[1])

            # グラフを更新
            line_3000.set_data(time_data, amplitude_data_3000)
            line_5000.set_data(time_data, amplitude_data_5000)
            ax.set_xlim(max(0, elapsed_time - 2), elapsed_time)  # 表示範囲を動的に調整
            ax.set_ylim(0, max(5000, max(amplitude_data_3000 + amplitude_data_5000) * 1.1))  # 振幅範囲を動的に調整
            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("終了します...")

# ストリームを停止し、PyAudioを終了
stream.stop_stream()
stream.close()
p.terminate()
