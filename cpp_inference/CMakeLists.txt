cmake_minimum_required(VERSION 3.15)
project(OpenVINODenoiserEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV Package
find_package(OpenCV REQUIRED)

# Find OpenVINO Package
find_package(OpenVINO REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(openvino_denoiser
    src/main.cpp
)

target_link_libraries(openvino_denoiser
    PRIVATE
    ${OpenCV_LIBS}
    openvino::runtime
)

message(STATUS "Build Configuration Complete for OpenVINO C++ Inference Engine.")

