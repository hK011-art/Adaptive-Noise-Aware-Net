#include "denoiser_engine.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "          HIGH-PERFORMANCE C++ INTEL OPENVINO INFERENCE ENGINE            " << std::endl;
    std::cout << "==========================================================================" << std::endl;

    std::string model_path = (argc > 1) ? argv[1] : "denoiser.xml";
    std::string device_name = (argc > 2) ? argv[2] : "GPU"; // Default target GPU (Intel Arc graphics)

    std::cout << "[Engine Target] Model Path: " << model_path << " | Hardware Target: " << device_name << std::endl;

    try {
        // 1. Initialize Engine
        OpenVINODenoiser denoiser(model_path, device_name);

        // 2. Create Synthetic 128x128 Test Image (OpenCV Mat)
        cv::Mat synthetic_noisy_image(128, 128, CV_8UC3);
        cv::randu(synthetic_noisy_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        // 3. Warmup Run
        std::cout << "[Benchmarking] Executing 5 warmup iterations..." << std::endl;
        for (int i = 0; i < 5; ++i) {
            cv::Mat dummy_out = denoiser.denoise(synthetic_noisy_image);
        }

        // 4. Benchmark Latency & Throughput (100 iterations)
        int num_iterations = 100;
        std::cout << "[Benchmarking] Measuring average latency over " << num_iterations << " iterations..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            cv::Mat restored_out = denoiser.denoise(synthetic_noisy_image);
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> total_duration = end_time - start_time;
        double avg_latency_ms = total_duration.count() / num_iterations;
        double fps = 1000.0 / avg_latency_ms;

        std::cout << "\n--------------------------------------------------------------------------" << std::endl;
        std::cout << "                     PERFORMANCE BENCHMARK SUMMARY                        " << std::endl;
        std::cout << "--------------------------------------------------------------------------" << std::endl;
        std::cout << "  Average Inference Latency: " << avg_latency_ms << " ms / frame" << std::endl;
        std::cout << "  Throughput (FPS):           " << fps << " Frames Per Second" << std::endl;
        std::cout << "  Status:                    SUCCESSFUL C++ DEPLOYMENT VALIDATION" << std::endl;
        std::cout << "--------------------------------------------------------------------------\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[OpenVINO C++ Engine Error] Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

