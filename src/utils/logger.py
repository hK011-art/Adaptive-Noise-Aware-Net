import os
import json
import numpy as np

class ExperimentLogger:
    """
    Logs metrics, loss curves, and performs statistical aggregation (mean ± std) across multiple random seeds.
    """
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, f"{experiment_name}_log.json")
        os.makedirs(log_dir, exist_ok=True)
        self.history = {"epochs": [], "train_loss": [], "val_psnr": [], "val_ssim": [], "val_lpips": []}

    def log_epoch(self, epoch: int, train_loss: float, psnr: float, ssim: float, lpips: float):
        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_psnr"].append(psnr)
        self.history["val_ssim"].append(ssim)
        self.history["val_lpips"].append(lpips)

        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    @staticmethod
    def aggregate_seeds(results_list: list) -> dict:
        """
        Takes a list of metric dicts across random seeds e.g., [{'psnr': 34.5, 'ssim': 0.91}, ...]
        and returns mean ± std for each metric.
        """
        aggregated = {}
        if not results_list:
            return aggregated

        keys = results_list[0].keys()
        for k in keys:
            vals = [res[k] for res in results_list if k in res]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            aggregated[k] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "formatted": f"{mean_val:.2f} ± {std_val:.3f}" if "psnr" in k.lower() else f"{mean_val:.4f} ± {std_val:.4f}"
            }
        return aggregated

