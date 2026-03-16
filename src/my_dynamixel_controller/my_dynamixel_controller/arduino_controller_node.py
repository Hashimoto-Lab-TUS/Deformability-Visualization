import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
import pandas as pd

class ArduinoController(Node):
    def __init__(self):
        super().__init__('arduino_controller')
        self.subscription = self.create_subscription(
            String,
            'csv_saved',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        csv_path = msg.data
        self.get_logger().info(f"CSV受信: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            material = df[df.iloc[:, 0] == 'Predicted Material'].iloc[0, 1]
            hardness = df[df.iloc[:, 0] == 'Object Hardness'].iloc[0, 1]
            self.get_logger().info(f"素材: {material}, 硬さ: {hardness}")
        except Exception as e:
            self.get_logger().error(f"CSV読み込みエラー: {e}")
            return

        # Arduinoと通信
        try:
            arduino_port = "/dev/ttyUSB0"  # または "COM4"（Windows）
            baud_rate = 115200
            ser = serial.Serial(arduino_port, baud_rate, timeout=1)
            time.sleep(2)

            self.get_logger().info("Arduinoに信号送信中...")
            ser.write(b'1')  # 必要に応じて "1" を素材や硬さに応じて変更してもOK
            time.sleep(1)

            while ser.in_waiting > 0:
                line = ser.readline().decode().strip()
                self.get_logger().info(f"Arduino応答: {line}")

            ser.close()
        except Exception as e:
            self.get_logger().error(f"Arduino通信エラー: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArduinoController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
