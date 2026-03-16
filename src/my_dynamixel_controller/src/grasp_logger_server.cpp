#include <rclcpp/rclcpp.hpp>
#include "my_dynamixel_controller/srv/dynamixel_log.hpp"  // サービスヘッダー
#include <dynamixel_sdk/dynamixel_sdk.h>                 // Dynamixel SDK C++版
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <cmath>

using namespace std::chrono_literals;
using DynamixelLog = my_dynamixel_controller::srv::DynamixelLog;

class GraspLoggerServer : public rclcpp::Node
{
public:
  GraspLoggerServer() : Node("grasp_logger_server")
  {
    service_ = this->create_service<DynamixelLog>(
      "dynamixel_log",
      std::bind(&GraspLoggerServer::handle_log_request, this, std::placeholders::_1, std::placeholders::_2));
    
    RCLCPP_INFO(this->get_logger(), "Grasp Logger Service Ready");
  }

private:
  rclcpp::Service<DynamixelLog>::SharedPtr service_;

  // Dynamixel control table addresses (例)
  static constexpr uint16_t ADDR_TORQUE_ENABLE = 64;
  static constexpr uint16_t ADDR_OPERATING_MODE = 11;
  static constexpr uint16_t ADDR_GOAL_POSITION = 116;
  static constexpr uint16_t ADDR_PRESENT_POSITION = 132;
  static constexpr uint16_t ADDR_CURRENT_LIMIT = 102;
  static constexpr uint16_t ADDR_PRESENT_CURRENT = 126;
  static constexpr uint16_t ADDR_PROFILE_VELOCITY = 112;

  static constexpr uint8_t PROTOCOL_VERSION = 2;
  static constexpr uint8_t DXL_ID = 1;
  static constexpr int BAUDRATE = 57600;
  static constexpr const char * DEVICENAME = "/dev/ttyUSB0"; // 要環境に合わせて変更

  static constexpr uint8_t TORQUE_ENABLE = 1;
  static constexpr uint8_t TORQUE_DISABLE = 0;
  static constexpr uint8_t CURRENT_BASED_POSITION_CONTROL = 5;
  static constexpr int DXL_MOVING_STATUS_THRESHOLD = 25;

  static constexpr int MAX_CURRENT_LIMIT = 250; // mA
  static constexpr int PROFILE_VELOCITY = 10;

  void handle_log_request(
    const std::shared_ptr<DynamixelLog::Request> request,
    std::shared_ptr<DynamixelLog::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), "Received request: initial_gripper_distance=%.2f, object_diameter=%.2f",
      request->initial_gripper_distance, request->object_diameter);

    // PortHandler, PacketHandler初期化
    dynamixel::PortHandler * portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    dynamixel::PacketHandler * packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    if (!portHandler->openPort()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open port");
      response->csv_filename = "ERROR: Failed to open port";
      response->graph_filename = "";
      return;
    }
    if (!portHandler->setBaudRate(BAUDRATE)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set baudrate");
      response->csv_filename = "ERROR: Failed to set baudrate";
      response->graph_filename = "";
      portHandler->closePort();
      return;
    }

    // 動作モード設定
    auto dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, CURRENT_BASED_POSITION_CONTROL);
    if (dxl_comm_result != COMM_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set operating mode: %d", dxl_comm_result);
    } else {
      RCLCPP_INFO(this->get_logger(), "Current limit set to %d mA", MAX_CURRENT_LIMIT);
    }

    // トルク有効化
    dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE);
    if (dxl_comm_result != COMM_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to enable torque: %d", dxl_comm_result);
    }

    // 移動距離計算
    double movement_distance = (request->initial_gripper_distance - request->object_diameter + 10);
    int movement_position = static_cast<int>(movement_distance * 30);

    // 現在位置取得
    uint32_t present_position = 0;
    uint8_t dxl_error = 0;
    dxl_comm_result = packetHandler->read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION, &present_position, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to read present position");
    }

    int dxl_goal_position = present_position + movement_position;

    // 電流制限設定
    dxl_comm_result = packetHandler->write2ByteTxRx(portHandler, DXL_ID, ADDR_CURRENT_LIMIT, MAX_CURRENT_LIMIT);

    // プロファイル速度設定
    dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, DXL_ID, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY);

    // 目標位置設定
    dxl_comm_result = packetHandler->write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, dxl_goal_position);

    // ログ記録用の変数
    std::vector<double> times, currents;
    std::vector<uint32_t> positions;

    auto start = std::chrono::steady_clock::now();
    
    rclcpp::Rate rate(100);  // 100 Hzで記録

    while (true) {
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - start).count();
      if (elapsed >= 10.0) break;

      // 電流値取得 (16bit符号付き変換)
      uint16_t present_current_raw = 0;
      dxl_comm_result = packetHandler->read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_CURRENT, &present_current_raw, &dxl_error);
      int present_current = (present_current_raw > 32767) ? (present_current_raw - 65536) : present_current_raw;

      // 位置取得
      uint32_t present_position_tmp = 0;
      dxl_comm_result = packetHandler->read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION, &present_position_tmp, &dxl_error);

      times.push_back(elapsed);
      currents.push_back(present_current);
      positions.push_back(present_position_tmp);

      RCLCPP_INFO(this->get_logger(), "Time: %.2f sec, Current: %d mA, Position: %u",
                  elapsed, present_current, present_position_tmp);

      // 目標到達判定
      if (std::abs(dxl_goal_position - static_cast<int>(present_position_tmp)) < DXL_MOVING_STATUS_THRESHOLD) {
        RCLCPP_INFO(this->get_logger(), "Reached target position. Logging for 1 more second...");

        auto end_time = now + std::chrono::seconds(1);
        while (std::chrono::steady_clock::now() < end_time) {
          auto now2 = std::chrono::steady_clock::now();
          double elapsed2 = std::chrono::duration<double>(now2 - start).count();

          dxl_comm_result = packetHandler->read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_CURRENT, &present_current_raw, &dxl_error);
          present_current = (present_current_raw > 32767) ? (present_current_raw - 65536) : present_current_raw;

          dxl_comm_result = packetHandler->read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION, &present_position_tmp, &dxl_error);

          times.push_back(elapsed2);
          currents.push_back(present_current);
          positions.push_back(present_position_tmp);

          RCLCPP_INFO(this->get_logger(), "Time: %.2f sec, Current: %d mA, Position: %u",
                      elapsed2, present_current, present_position_tmp);

          std::this_thread::sleep_for(10ms);
        }
        break;
      }
      std::this_thread::sleep_for(10ms);
    }

    // CSV保存
    std::string csv_filename = "current_position_data.csv";
    std::ofstream ofs(csv_filename);
    ofs << "Time (s),Current (mA),Position\n";
    for (size_t i = 0; i < times.size(); i++) {
      ofs << times[i] << "," << currents[i] << "," << positions[i] << "\n";
    }
    ofs.close();

    // トルク解除してポート閉じる
    packetHandler->write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE);
    portHandler->closePort();

    response->csv_filename = csv_filename;
    response->graph_filename = "NOT GENERATED"; // グラフは別途Pythonなどで生成推奨

    RCLCPP_INFO(this->get_logger(), "CSV saved: %s", csv_filename.c_str());
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GraspLoggerServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
