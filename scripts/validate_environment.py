import sys
import importlib

def validate_environment():
    print("============================================================")
    print("        RESEARCH ENVIRONMENT VERIFICATION SUITE             ")
    print("============================================================")
    
    blocked = False
    
    # 1. Python version check
    py_ver = sys.version_info
    print(f"Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 9):
        print("  [ERROR] Python >= 3.9 is required!")
        blocked = True
    else:
        print("  [PASS] Python version compatible.")

    # 2. PyTorch & Acceleration checks
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            print(f"  [PASS] CUDA Available: {dev_name} (Compute Capability: {torch.cuda.get_device_capability(0)})")
        else:
            print("  [NOTICE] CUDA not detected. CPU backend operational (Unit tests & smoke tests supported).")
    except ImportError:
        print("  [ERROR] PyTorch is not installed!")
        blocked = True

    # 3. Required scientific packages
    required_packages = [
        ("numpy", "Core numerical computing"),
        ("scipy", "Statistical tests (Wilcoxon)"),
        ("PIL", "Pillow image processing"),
        ("yaml", "PyYAML configuration management"),
        ("pandas", "Metric export & tabular statistics"),
    ]

    for pkg_name, purpose in required_packages:
        try:
            mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [PASS] {pkg_name:<10} ({ver}): {purpose}")
        except ImportError:
            print(f"  [ERROR] Missing required package: {pkg_name} ({purpose})")
            blocked = True

    # 4. Optional / Recommended packages
    optional_packages = [
        ("lpips", "Perceptual LPIPS metric computation"),
        ("openvino", "Intel OpenVINO edge inference runtime"),
        ("onnx", "ONNX model representation & export"),
        ("onnxruntime", "ONNX execution engine"),
    ]

    for pkg_name, purpose in optional_packages:
        try:
            mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [INFO] {pkg_name:<10} ({ver}): {purpose}")
        except ImportError:
            print(f"  [NOTICE] Optional package not installed: {pkg_name} ({purpose})")

    print("============================================================")
    if blocked:
        print("  >>> OVERALL STATUS: ENVIRONMENT BLOCKED <<<")
        print("  Please install missing packages before initiating training.")
        sys.exit(1)
    else:
        print("  >>> OVERALL STATUS: ENVIRONMENT OK <<<")
        print("  Ready for reproducible CPU/GPU ML pipeline execution.")
    print("============================================================\n")

if __name__ == "__main__":
    validate_environment()

