from collections import Counter

import numpy as np
from torch.utils.data import Sampler


class UnderSampler(Sampler): # type: ignore[type-arg]
    def __init__(self, labels, majority_class, majority_fraction=0.2): # type: ignore[no-untyped-def]
        self.labels = labels
        self.majority_class = majority_class
        self.majority_fraction = majority_fraction
        self.indices = self._get_balanced_indices()

    def _get_balanced_indices(self):  # type: ignore[no-untyped-def]
        label_counts = Counter([tuple(label.tolist()) for label in self.labels])
        majority_count = label_counts[tuple(self.majority_class)]
        majority_sample_size = int(majority_count * self.majority_fraction)

        class_indices = {cls: [] for cls in label_counts.keys()}
        for idx, label in enumerate(self.labels):
            class_indices[tuple(label.tolist())].append(idx)

        balanced_indices = []
        for cls, indices in class_indices.items():
            if cls == tuple(self.majority_class):
                balanced_indices.extend(np.random.choice(indices, majority_sample_size, replace=False))
            else:
                balanced_indices.extend(indices)

        np.random.shuffle(balanced_indices)
        return balanced_indices

    def __iter__(self): # type: ignore[no-untyped-def]
        return iter(self.indices)

    def __len__(self): # type: ignore[no-untyped-def]
        return len(self.indices)
