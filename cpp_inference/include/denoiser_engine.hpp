#ifndef DENOISER_ENGINE_HPP
#define DENOISER_ENGINE_HPP

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

class OpenVINODenoiser {
private:
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::string device_name;

public:
    OpenVINODenoiser(const std::string& model_path, const std::string& target_device = "CPU")
        : device_name(target_device) {
        std::cout << "[OpenVINO Engine] Loading Model: " << model_path << " on Device: " << target_device << std::endl;
        
        // Load Model into OpenVINO Core
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        
        // Compile Model for Target Hardware (GPU/GPU.0 for Intel Arc graphics, CPU fallback)
        try {
            compiled_model = core.compile_model(model, device_name);
        } catch (const std::exception& e) {
            std::cout << "[OpenVINO Warning] Failed to compile on " << device_name << " (" << e.what() << "). Falling back to CPU..." << std::endl;
            compiled_model = core.compile_model(model, "CPU");
        }
        
        infer_request = compiled_model.create_infer_request();
        std::cout << "[OpenVINO Engine] Initialization Complete." << std::endl;
    }

    cv::Mat denoise(const cv::Mat& input_bgr) {
        // Preprocessing: BGR -> RGB [0.0, 1.0] NCHW layout
        cv::Mat input_rgb;
        cv::cvtColor(input_bgr, input_rgb, cv::COLOR_BGR2RGB);
        input_rgb.convertTo(input_rgb, CV_32FC3, 1.0 / 255.0);

        int height = input_rgb.rows;
        int width = input_rgb.cols;

        // Allocate OpenVINO Tensor
        ov::Tensor input_tensor(ov::element::f32, {1, 3, static_cast<size_t>(height), static_cast<size_t>(width)});
        float* tensor_data = input_tensor.data<float>();

        // Convert HWC (OpenCV) to NCHW (PyTorch/OpenVINO)
        std::vector<cv::Mat> channels(3);
        for (int c = 0; c < 3; ++c) {
            channels[c] = cv::Mat(height, width, CV_32FC1, tensor_data + c * height * width);
        }
        cv::split(input_rgb, channels);

        // Run Inference
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        // Retrieve Output Tensor [1, 3, H, W]
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        const float* out_data = output_tensor.data<const float>();

        // Convert NCHW back to HWC RGB -> BGR [0, 255] uint8
        std::vector<cv::Mat> out_channels(3);
        for (int c = 0; c < 3; ++c) {
            out_channels[c] = cv::Mat(height, width, CV_32FC1, const_cast<float*>(out_data + c * height * width));
        }

        cv::Mat restored_rgb;
        cv::merge(out_channels, restored_rgb);
        cv::Mat restored_bgr;
        cv::cvtColor(restored_rgb, restored_bgr, cv::COLOR_RGB2BGR);
        restored_bgr.convertTo(restored_bgr, CV_8UC3, 255.0);

        return restored_bgr;
    }
};

#endif // DENOISER_ENGINE_HPP

