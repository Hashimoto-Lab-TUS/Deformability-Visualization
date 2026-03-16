import rclpy
import pandas as pd
from rclpy.node import Node
from my_dynamixel_controller.srv import DynamixelLog  # .srvに合わせてパッケージ名調整
from std_msgs.msg import String

class DynamixelLogClient(Node):

    def __init__(self):
        super().__init__('dynamixel_log_client')
        self.client = self.create_client(DynamixelLog, 'dynamixel_log')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('サービス待機中...')

        # self.publisher_ は __init__ で定義
        self.publisher_ = self.create_publisher(String, 'csv_saved', 10)

        # ユーザーからの入力を取得
        self.initial = float(input("初期グリッパー距離を入力してください（例: 50.0）: "))
        self.diameter = float(input("物体の直径を入力してください（例: 30.0）: "))

        # リクエスト作成
        request = DynamixelLog.Request()
        request.initial_gripper_distance = self.initial
        request.object_diameter = self.diameter

        # サービス呼び出し
        future = self.client.call_async(request)
        future.add_done_callback(self.response_callback)
        rclpy.spin_until_future_complete(self, future)
        #rclpy.spin(self, future)

        if future.result() is not None:
            self.get_logger().info(f"CSV: {future.result().csv_filename}")
            self.get_logger().info(f"Graph: {future.result().graph_filename}")
        else:
            self.get_logger().error('サービス呼び出しに失敗しました')
    
    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"CSVファイルを受け取りました：{response.csv_filename}")
            self.get_logger().info(f"Graphファイル名：{response.graph_filename}")
            initial_gripper_distance = self.initial
            object_diameter = self.diameter

            input_csv = response.csv_filename
            output_csv = 'converted_data.csv'

            def convert_position_to_gripper(pos, initial_motor_position):
                return initial_gripper_distance - (pos - initial_motor_position) / 40

            def convert_current_to_force(current):
                return (0.1716 * current - 1.642)

            df = pd.read_csv(input_csv)
            initial_motor_position = df['Position'].iloc[0]

            df['Gripper Position (mm)'] = df['Position'].apply(
                lambda pos: convert_position_to_gripper(pos, initial_motor_position))
            df['Force (N)'] = df['Current (mA)'].apply(convert_current_to_force)

            filtered_df = df[df['Gripper Position (mm)'] < object_diameter]

            force_average = filtered_df['Force (N)'].mean() if not filtered_df.empty else None

            if not filtered_df.empty:
                final_gripper_position = filtered_df['Gripper Position (mm)'].iloc[-1]
                displacement = object_diameter - final_gripper_position
            else:
                displacement = None

            if displacement and displacement != 0:
                stiffness = force_average / displacement
            else:
                stiffness = None

            df.at[0, 'Average Force (N)'] = force_average
            df.at[0, 'Displacement (mm)'] = displacement
            df.at[0, 'Stiffness (N/mm)'] = stiffness

            df['Average Force (N)'] = df['Average Force (N)'].fillna("")
            df['Displacement (mm)'] = df['Displacement (mm)'].fillna("")
            df['Stiffness (N/mm)'] = df['Stiffness (N/mm)'].fillna("")

            df.to_csv(output_csv, index=False)

            self.get_logger().info(f"変換が完了しました。結果は {output_csv} に保存されています。")

            # === 剛性に基づく素材スコア計算 ===

            # ファイルパス
            probabilities_csv = r"C:\home\reo\dev\Multimodal\probabilities.csv"
            stiffness_csv = r"C:/home/reo/dev/Multimodal/converted_data.csv"
            output_csv = r"C:\home\reo\dev\Multimodal\final_scores.csv"

            # 剛性定義
            material_stiffness_definitions = {
            "paper": 3.148,
            "soft_plastic": 2.504,
            "hard_plastic": 36.64,
            "glass": 71.46,
            "metal": 49.64
            }

            soft_materials = ["paper", "soft_plastic"]
            hard_materials = ["hard_plastic", "glass", "metal"]

            # CSV読み込み
            prob_df = pd.read_csv(probabilities_csv)
            stiff_df = pd.read_csv(stiffness_csv)
            calculated_stiffness = stiff_df.iloc[0]["Stiffness (N/mm)"]

            # 確率読み取り（soft/hard plasticに分ける）
            probabilities = {
            "glass": prob_df.iloc[0, 1],
            "metal": prob_df.iloc[0, 2],
            "paper": prob_df.iloc[0, 3],
            "soft_plastic": prob_df.iloc[0, 4],
            "hard_plastic": prob_df.iloc[0, 4]
            }

            # スコア計算関数
            def calculate_scores(probabilities, calculated_stiffness, threshold=0.09):
                scores = {}
                for material, prob in probabilities.items():
                    if prob >= threshold:
                        defined_stiffness = material_stiffness_definitions.get(material, 0)
                        score = abs(calculated_stiffness - defined_stiffness) * (1 - prob)
                        scores[material] = score
                return scores

            # スコア計算
            scores = calculate_scores(probabilities, calculated_stiffness)

            # 最小スコア素材を判定
            predicted_material = min(scores, key=scores.get)
            object_hardness = "Soft" if calculated_stiffness <= 10 else "Hard"

            # 結果保存
            result_df = pd.DataFrame({
                "Material": scores.keys(),
                "Score": scores.values()
            })

            # 不要な素材を除去
            if object_hardness == "Soft":
                result_df = result_df[~result_df["Material"].isin(hard_materials)]
            elif object_hardness == "Hard":
                result_df = result_df[~result_df["Material"].isin(soft_materials)]

            # 最終結果追加
            result_df.loc[len(result_df) + 3] = ["Predicted Material", predicted_material]
            result_df.loc[len(result_df) + 4] = ["Object Hardness", object_hardness]

            # 保存
            result_df.to_csv(output_csv, index=False)
            self.get_logger().info(f"スコアを {output_csv} に保存しました")
            self.get_logger().info(f"素材判定結果：{predicted_material}、硬さ：{object_hardness}")

            # publish トピック
            msg = String()
            msg.data = output_csv
            self.publisher_.publish(msg)
            self.get_logger().info(f"保存したCSVファイル名を通知しました: {output_csv}")
            
            #rclpy.shutdown() 
        except Exception as e:
            self.get_logger().error(f"サービス呼び出しに失敗しちゃった:{e}")
        finally:
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    client = DynamixelLogClient()
    rclpy.spin(client)
    #rclpy.shutdown()

if __name__ == '__main__':
    main()
