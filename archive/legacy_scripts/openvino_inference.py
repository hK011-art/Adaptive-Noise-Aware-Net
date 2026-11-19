import os
import time
import argparse
import numpy as np

def run_inference(model_path="adaptive_denoiser.pt", device="AUTO"):
    print("==========================================================================")
    print("           HIGH-PERFORMANCE INFERENCE & LATENCY BENCHMARK                 ")
    print("==========================================================================")
    print(f"Loading Model:   {model_path}")
    print(f"Target Device:   {device}")

    # Check for OpenVINO
    has_openvino = False
    try:
        import openvino as ov
        has_openvino = True
    except ImportError:
        pass

    # Prepare 128x128 synthetic test image tensor [1, 3, 128, 128]
    input_np = np.random.rand(1, 3, 128, 128).astype(np.float32)

    if has_openvino:
        core = ov.Core()
        available = core.available_devices
        print(f"OpenVINO Engine: Active | Available Devices: {available}")
        
        target = "GPU" if ("GPU" in available and device in ["GPU", "AUTO"]) else ("CPU" if "CPU" in available else available[0])
        print(f"Compiling on Hardware Accelerator: {target}...")
        
        model = core.read_model(model_path)
        compiled_model = core.compile_model(model, target)
        infer_request = compiled_model.create_infer_request()
        
        # Warmup
        for _ in range(5):
            _ = infer_request.infer({0: input_np})
            
        num_frames = 100
        print(f"Benchmarking inference latency over {num_frames} frames...")
        
        start = time.perf_counter()
        for _ in range(num_frames):
            _ = infer_request.infer({0: input_np})
        end = time.perf_counter()
        
        engine_name = f"Intel OpenVINO Runtime ({target})"
    else:
        # PyTorch TorchScript JIT Engine Fallback
        import torch
        print("OpenVINO Python runtime not detected. Executing with TorchScript JIT Engine...")
        
        torch_device = torch.device("cuda" if (torch.cuda.is_available() and device != "CPU") else "cpu")
        model = torch.jit.load(model_path, map_location=torch_device)
        model.eval()
        
        input_tensor = torch.from_numpy(input_np).to(torch_device)
        
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
                
            num_frames = 100
            print(f"Benchmarking inference latency over {num_frames} frames on {torch_device}...")
            
            start = time.perf_counter()
            for _ in range(num_frames):
                _ = model(input_tensor)
            end = time.perf_counter()
            
        engine_name = f"TorchScript JIT Engine ({torch_device})"

    total_time_ms = (end - start) * 1000.0
    avg_latency_ms = total_time_ms / num_frames
    fps = 1000.0 / avg_latency_ms

    print("\n--------------------------------------------------------------------------")
    print("                     PERFORMANCE BENCHMARK SUMMARY                        ")
    print("--------------------------------------------------------------------------")
    print(f"  Inference Engine:          {engine_name}")
    print(f"  Average Latency per Frame: {avg_latency_ms:.2f} ms")
    print(f"  Throughput:                {fps:.2f} FPS (Frames Per Second)")
    print("--------------------------------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="adaptive_denoiser.pt", help="Path to TorchScript or ONNX model")
    parser.add_argument("--device", type=str, default="AUTO", choices=["CPU", "GPU", "AUTO"], help="Target device")
    args = parser.parse_args()

    run_inference(model_path=args.model, device=args.device)
