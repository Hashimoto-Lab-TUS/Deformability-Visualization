import pyaudio
import wave

# 録音の設定
FORMAT = pyaudio.paInt16  # 16ビットフォーマット
CHANNELS = 1  # モノラル録音
RATE = 44100  # サンプリング周波数 (44.1kHz)
CHUNK = 1024  # バッファサイズ
RECORD_SECONDS = int(input("録音時間を入力してください (秒): "))
OUTPUT_FILENAME = input("保存するファイル名を入力してください (例: output.wav): ")

# PyAudioの初期化
p = pyaudio.PyAudio()

# ストリームの開始
print("録音を開始します...")
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

frames = []

# 録音の実行
try:
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("録音が終了しました。")
except KeyboardInterrupt:
    print("\n録音を中断しました。")

# ストリームの停止と終了
stream.stop_stream()
stream.close()
p.terminate()

# 録音データをファイルに保存
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"録音した音声を {OUTPUT_FILENAME} に保存しました。")
