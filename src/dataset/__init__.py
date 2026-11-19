from .div2k_dataset import PhotographicDIV2KDataset
from .sidd_dataset import GenuineSIDDDataset
from .dnd_dataset import HeldOutDNDDataset

# Aliases for compatibility
SyntheticDIV2KDataset = PhotographicDIV2KDataset
SIDDDataset = GenuineSIDDDataset
DNDDataset = HeldOutDNDDataset

__all__ = [
    "PhotographicDIV2KDataset",
    "GenuineSIDDDataset",
    "HeldOutDNDDataset",
    "SyntheticDIV2KDataset",
    "SIDDDataset",
    "DNDDataset"
]
