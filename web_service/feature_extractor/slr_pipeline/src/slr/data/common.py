"""Functionality that is common to all datasets, for example, collecting the list of samples."""
import csv
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Sample:
    path: str
    label: int
    frame_indices: Optional[List[int]]


def keep_class(label: int, topk_classes: int) -> bool:
    """We keep a class if its `label` is smaller than `topk_classes`, or if `topk_classes` is smaller than 2.
    Assumes that the labels are ordered according to descending class occurrence."""
    return topk_classes <= 1 or label < topk_classes


def collect_samples(root_path: str, job: str, samples_file_override: Optional[str]) -> List[Sample]:
    samples = []
    if samples_file_override is not None:
        samples_filename = samples_file_override
    else:
        samples_filename = 'samples.csv'
    with open(os.path.join(root_path, samples_filename), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ['Id', 'Label', 'Participant', 'Video', 'Subset']
        for row in reader:
            _id, label, _participant, video, subset = row
            if subset == job:
                samples.append(Sample(video, int(label), None))
    return samples
