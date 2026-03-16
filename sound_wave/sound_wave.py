import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# マイクの設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 #サンプリング周波数 44100 Hz
CHUNK = 1024

# ファイルに保存する設定
name = input("Enter material name:")
output_file = name + "_frequency_data.csv"

# PyAudioオブジェクトを作成
p = pyaudio.PyAudio()

# ストリームを開始
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("マイク入力を開始...")

# グラフの設定
plt.ion()
fig, ax = plt.subplots()
x = np.fft.fftfreq(CHUNK, 1 / RATE)[:CHUNK // 2]
line, = ax.plot(x, np.zeros(CHUNK // 2))

ax.set_ylim(0, 5000)  # 振幅の範囲を調整
ax.set_xlim(0, RATE / 2)  # 周波数の範囲

# フラグを設定して終了を制御
is_running = True

def on_close(event):
    """グラフウィンドウが閉じられたときに実行される関数"""
    global is_running
    is_running = False

fig.canvas.mpl_connect('close_event', on_close)

# 周波数データの保存準備
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Frequency", "Amplitude"])  # ヘッダー行を追加

    start_time = time.time()

    try:
        while is_running:  # ウィンドウが開いている間だけ実行
            # 音声データを取得
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

            # FFTを計算
            fft_data = np.abs(np.fft.fft(data))[:CHUNK // 2]

            # ピーク周波数を計算
            peak_freq = x[np.argmax(fft_data)]
            peak_amp = np.max(fft_data)

            # 時間を記録
            elapsed_time = time.time() - start_time

            # 周波数データをファイルに保存
            writer.writerow([elapsed_time, peak_freq, peak_amp])

            # グラフを更新
            line.set_ydata(fft_data)
            fig.canvas.draw()
            fig.canvas.flush_events()

    except Exception as e:
        print(f"エラーが発生しました: {e}")

print("終了します...")

# ストリームを停止し、PyAudioを終了
stream.stop_stream()
stream.close()
p.terminate()
