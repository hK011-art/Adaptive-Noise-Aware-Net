import os
import sys
import shutil
import glob
from pathlib import Path

print("="*60)
print("  STEP 1: REPOSITORY SETUP")
print("="*60)

root = Path("/kaggle/working/cv_pro")
root.mkdir(parents=True, exist_ok=True)

# 1. Flatten if nested "CV pro" exists
nested = root / "CV pro"
if nested.exists():
    for item in nested.iterdir():
        dst = root / item.name
        if dst.exists():
            if dst.is_dir(): shutil.rmtree(dst)
            else: dst.unlink()
        shutil.move(str(item), str(root))
    nested.rmdir()

# 2. Copy from input if empty
src_input = Path("/kaggle/input/datasets/hk1131/cv-pro-zip/CV pro")
if not (root / "scripts").exists() and src_input.exists():
    shutil.copytree(src_input, root, dirs_exist_ok=True)

print("Repository root verified at:", root)

# 3. Create tests/fixtures/toy_data.py
fix_dir = root / "tests" / "fixtures"
fix_dir.mkdir(parents=True, exist_ok=True)
(fix_dir / "__init__.py").write_text("")
(fix_dir / "toy_data.py").write_text('''import torch
from torch.utils.data import Dataset
class ToySyntheticDataset(Dataset):
    def __init__(self, num_samples=16, patch_size=64, seed=42):
        super().__init__()
        self.samples = [(torch.rand(3, patch_size, patch_size), torch.rand(3, patch_size, patch_size)) for _ in range(num_samples)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
''')

# 4. Remove the broken import from train.py if present
train_file = root / "scripts" / "train.py"
if train_file.exists():
    lines = train_file.read_text().splitlines()
    new_lines = []
    for line in lines:
        if "from tests.fixtures.toy_data import ToySyntheticDataset" in line:
            new_lines.append("""try:
    from tests.fixtures.toy_data import ToySyntheticDataset
except Exception:
    class ToySyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=16, patch_size=64, seed=42):
            super().__init__()
            self.samples = [(torch.rand(3, patch_size, patch_size), torch.rand(3, patch_size, patch_size)) for _ in range(num_samples)]
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx): return self.samples[idx]""")
        else:
            new_lines.append(line)
    train_file.write_text("\n".join(new_lines))

print("Fixed train.py imports.")

# 5. Link datasets
data_dir = root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

div2k_dst = data_dir / "div2k"
if not div2k_dst.exists():
    for r, d, f in os.walk("/kaggle/input"):
        pngs = [x for x in f if x.lower().endswith(('.png', '.jpg'))]
        if len(pngs) >= 10 and ("div2k" in r.lower() or "isr" in r.lower() or "srgan" in r.lower()):
            os.symlink(r, div2k_dst)
            print("Linked DIV2K from:", r)
            break

sidd_dst = data_dir / "sidd"
if not sidd_dst.exists():
    for r, d, f in os.walk("/kaggle/input"):
        pngs = [x for x in f if x.lower().endswith(('.png', '.jpg'))]
        if len(pngs) >= 2 and ("sidd" in r.lower() or "smartphone" in r.lower()):
            os.symlink(r, sidd_dst)
            print("Linked SIDD from:", r)
            break

print("Datasets linked successfully.")
print("\n" + "="*60)
print("  READY TO TRAIN!")
print("="*60)

