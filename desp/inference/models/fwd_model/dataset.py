"""
Adapted from ASKCOSv2 template relevance module:
https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance/-/blob/main/dataset.py?ref_type=heads
"""

import misc
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple


def init_loader(args, dataset, batch_size: int, shuffle: bool = False):
    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_cores,
        pin_memory=True,
    )

    return loader


class FingerprintDataset(Dataset):
    """
    Dataset class for fingerprint representation of products
    for template relevance prediction
    """

    def __init__(self, fp_file: str, label_file: str, model_type: str):
        misc.log_rank_0(f"Loading pre-computed product fingerprints from {fp_file}")
        self.data = sparse.load_npz(fp_file)
        self.data = self.data.tocsr()
        self.model_type = model_type

        misc.log_rank_0(f"Loading pre-computed target labels from {label_file}")
        if self.model_type == "templ_rel":
            self.labels = np.load(label_file)
        elif self.model_type == "bb":
            self.labels = sparse.load_npz(label_file)
            self.labels = self.labels.tocsr()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        assert self.data.shape[0] == self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns tuple of product fingerprint, and label (template index)
        """
        fp = self.data[idx].toarray()
        if self.model_type == "templ_rel":
            label = self.labels[idx]
        elif self.model_type == "bb":
            label = self.labels[idx].toarray()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return fp, label

    def __len__(self) -> int:
        return self.data.shape[0]
