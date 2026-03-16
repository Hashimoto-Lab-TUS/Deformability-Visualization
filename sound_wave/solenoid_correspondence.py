import serial
import time

# Arduinoと接続するポートを指定（環境に応じて変更）
arduino_port = "COM4"  # Windowsの場合 (例: "COM3")
# arduino_port = "/dev/ttyUSB0"  # Linuxの場合
# arduino_port = "/dev/tty.usbmodemXXXX"  # macOSの場合 (XXXXは環境によって異なる)

baud_rate = 115200  # Arduinoのシリアル通信速度に合わせる

# シリアル通信を開始
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # Arduinoのリセット待ち

print("Sending signal to Arduino...")
ser.write(b'1')  # '1' を送信してソレノイドを制御
time.sleep(1)  # 少し待機

# Arduinoからのメッセージを表示
while ser.in_waiting > 0:
    print(ser.readline().decode().strip())

ser.close()  # シリアル通信を閉じる
