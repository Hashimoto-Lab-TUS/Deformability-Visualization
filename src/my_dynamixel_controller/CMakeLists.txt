cmake_minimum_required(VERSION 3.8)
project(my_dynamixel_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)   
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED) 
find_package(dynamixel_sdk REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DynamixelLog.srv"
)

ament_export_dependencies(rosidl_default_runtime)

# 実行ファイルの追加（例: grasp_logger_server.cpp をビルド）
add_executable(grasp_logger_server src/grasp_logger_server.cpp)

ament_target_dependencies(grasp_logger_server
  rclcpp
  rosidl_default_runtime
  dynamixel_sdk
)

rosidl_target_interfaces(grasp_logger_server
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

install(TARGETS
  grasp_logger_server
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Pythonスクリプトのインストール
install(PROGRAMS
  my_dynamixel_controller/call_logger_client.py  # 実際のクライアントPythonスクリプトのパス
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

