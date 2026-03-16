import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile
from scipy.signal import spectrogram

# 1. 音声データを読み込む
file_path = r"C:\Users\reooo\Desktop\dev\sound_wave\bbb"  # 収録済みの音響データファイル
rate, data = wavfile.read(file_path)

# モノラル対応
if len(data.shape) > 1:
    data = data[:, 0]  # 左チャンネルのみ使用

# 2. 振幅の計算（絶対値を取る）
time = np.linspace(0, len(data) / rate, num=len(data))
amplitude = np.abs(data)

# 3. ウェーブレット変換を実行
#wavelet = 'cmor'  # メキシカンハットウェーブレット
#scales = np.arange(1, 128)  # スケール範囲
#coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=1/rate)

# 4. FFT スペクトログラム
f, t, Sxx = spectrogram(data, fs=rate, nperseg=1024)

# 5. 可視化
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# 振幅 vs. 時間（x 軸を統一）
axes[0].plot(time, amplitude / 1000, color='black')
axes[0].set_title('Amplitude over Time')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_xlim(0, len(data) / rate)  # ここで修正

# FFT スペクトログラム（時間軸の統一）
im1 = axes[1].imshow(10 * np.log10(Sxx), aspect='auto', origin='lower', 
                      extent=[0, len(data) / rate, f[0] / 1000, f[-1] / 1000], cmap='inferno')
axes[1].set_title('FFT Spectrogram')
axes[1].set_ylabel('Frequency (kHz)')
axes[1].set_xlabel('Time (s)')

# ウェーブレット変換の結果（時間軸の統一）
#im2 = axes[2].imshow(np.abs(coefficients), extent=[0, len(data) / rate, frequencies[-1] / 1000, frequencies[0] / 1000],
                      #aspect='auto', cmap='jet')
#axes[2].set_title('Wavelet Transform')
#axes[2].set_ylabel('Frequency (kHz)')
#axes[2].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
