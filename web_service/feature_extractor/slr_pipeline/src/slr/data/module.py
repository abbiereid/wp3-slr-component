from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .pose import RawPoseDataset, CleanedPoseDataset, PoseInferenceDataset
from .rgb_full import FullFrameInferenceDataset, FullFrameDataset


def get_inference_data_loader(**kwargs):
    batch_size = kwargs['batch_size']
    data_kind = kwargs['data_kind']
    num_workers = kwargs['num_workers']
    data_dir = kwargs['data_dir']
    video_file_extension = kwargs["video_file_extension"]

    if data_kind.startswith('Mediapipe'):
        dataset = PoseInferenceDataset(data_dir, data_kind, video_file_extension, dict(kwargs))
    elif data_kind == 'RGB_Full':
        dataset = FullFrameInferenceDataset(data_dir, data_kind, '.mp4', dict(kwargs))
    else:
        raise ValueError(f'Unsupported data_kind {data_kind}.')

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      shuffle=True, collate_fn=get_collate_fn(data_kind, variable_length=True, inference_mode=True))


class DataModule(pl.LightningDataModule):
    def __init__(self, num_workers: int, batch_size: int, **kwargs):
        super(DataModule, self).__init__()

        # Arguments.
        self.data_kind = kwargs['data_kind']
        self.num_workers = num_workers
        self.data_dir = kwargs['data_dir']
        self.batch_size = batch_size
        self.variable_length_sequences = kwargs['variable_length_sequences']
        self.samples_file_override = kwargs['samples_file_override']

        # Initialization.
        if self.data_kind == 'Mediapipe_Raw':
            self.train_set = RawPoseDataset('train', 'mediapipe', **kwargs)
            self.val_set = RawPoseDataset('val', 'mediapipe', **kwargs)
            self.test_set = RawPoseDataset('test', 'mediapipe', **kwargs)
        elif self.data_kind.startswith('Mediapipe_Cleaned'):
            self.train_set = CleanedPoseDataset('train', 'mediapipe_post', **kwargs)
            self.val_set = CleanedPoseDataset('val', 'mediapipe_post', **kwargs)
            self.test_set = CleanedPoseDataset('test', 'mediapipe_post', **kwargs)
        elif self.data_kind == 'OpenPose_Raw':
            self.train_set = RawPoseDataset('train', 'openpose', **kwargs)
            self.val_set = RawPoseDataset('val', 'openpose', **kwargs)
            self.test_set = RawPoseDataset('test', 'openpose', **kwargs)
        elif self.data_kind.startswith('OpenPose_Cleaned'):
            self.train_set = CleanedPoseDataset('train', 'openpose_post', **kwargs)
            self.val_set = CleanedPoseDataset('val', 'openpose_post', **kwargs)
            self.test_set = CleanedPoseDataset('test', 'openpose_post', **kwargs)
        elif self.data_kind == 'MMPose_Raw':
            self.train_set = RawPoseDataset('train', 'mmpose', **kwargs)
            self.val_set = RawPoseDataset('val', 'mmpose', **kwargs)
            self.test_set = RawPoseDataset('test', 'mmpose', **kwargs)
        elif self.data_kind.startswith('MMPose_Cleaned'):
            self.train_set = CleanedPoseDataset('train', 'mmpose_post', **kwargs)
            self.val_set = CleanedPoseDataset('val', 'mmpose_post', **kwargs)
            self.test_set = CleanedPoseDataset('test', 'mmpose_post', **kwargs)
        elif self.data_kind == 'RGB_Full':
            self.train_set = FullFrameDataset(self.data_dir, 'train', self.samples_file_override)
            self.val_set = FullFrameDataset(self.data_dir, 'val', self.samples_file_override)
            self.test_set = FullFrameDataset(self.data_dir, 'test', self.samples_file_override)
        else:
            raise ValueError(f'Unknown dataset kind {self.data_kind}.')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          shuffle=True, collate_fn=get_collate_fn(self.data_kind, self.variable_length_sequences))

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=get_collate_fn(self.data_kind, self.variable_length_sequences))

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=get_collate_fn(self.data_kind, self.variable_length_sequences))

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument('--data_kind', type=str, help='The kind of data to load (e.g., RGB or keypoints).',
                            required=True)
        parser.add_argument('--num_workers', type=int, help='Number of DataLoader workers.', default=0)
        parser.add_argument('--data_dir', type=str, help='Root dataset directory.', required=True)
        parser.add_argument('--variable_length_sequences', action='store_true', help='Use variable length sequences.')
        parser.add_argument('--samples_file_override', type=str, help='Name of the samples file (default: samples.csv)')
        parser.add_argument('--data_path_override', type=str,
                            help='Useful for trying out different post-processing steps. See pose.py for usage.',
                            default='mediapipe_post')
        parser.add_argument('--mediapipe_2d', action='store_true',
                            help='Add this flag to drop the third dimension for MediaPipe keypoints.')
        # Augmentation flags for PTN.
        parser.add_argument('--augment_rotY', type=int,
                            help='If greater than zero, the maximum amount of degrees to rotate the pose with along the Y axis.',
                            default=0)
        parser.add_argument('--augment_hflip', action='store_true',
                            help='Enable random horizontal flipping with probability 0.5.')
        parser.add_argument('--augment_adaptSpeed', type=float,
                            help='Probability for adapting the speed augmentation. See also --augment_adaptSpeedFaster.',
                            default=0.0)
        parser.add_argument('--augment_adaptSpeedFaster', type=float,
                            help='Probability that adapting the speed will drop frames. If adapting the speed, but not dropping frames, will interpolate frames.',
                            default=0.5)
        parser.add_argument('--augment_temporalShift', type=float,
                            help='Probability for applying temporal shift augmentation.',
                            default=0.0)
        return parent_parser


class Batch:
    def batch_size(self):
        raise NotImplementedError('Implement me in the child class!')

    def to(self, device):
        raise NotImplementedError('Implement me in the child class!')


class InferenceBatch(Batch):
    def __init__(self, inputs: torch.Tensor, lengths: List[int], filenames: List[str]):
        self.inputs = inputs
        self.lengths = lengths
        self.filenames = filenames

    def batch_size(self):
        return self.inputs.size(1)  # batch_first = False.

    def to(self, device):
        self.inputs = self.inputs.to(device)
        return self


class FixedLengthBatch(Batch):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, filenames: List[str]):
        self.inputs = inputs
        self.targets = targets
        self.filenames = filenames

    def batch_size(self):
        return self.inputs.size(1)  # batch_first = False.

    def to(self, device):
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)
        return self


class VariableLengthBatch(Batch):
    def __init__(self, inputs: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor, filenames: List[str]):
        self.inputs = inputs
        self.lengths = lengths
        self.targets = targets
        self.filenames = filenames

    def batch_size(self):
        return self.inputs.size(1)  # batch_first = False.

    def to(self, device):
        self.inputs = self.inputs.to(device)
        # 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
        # self.lengths = self.lengths.to(device)
        self.targets = self.targets.to(device)
        return self


def get_collate_fn(data_kind: str, variable_length: bool, inference_mode: bool = False):
    def collate_images(batch):
        clips = [e[0] for e in batch]
        targets = [e[1] for e in batch]
        filenames = [e[2] for e in batch]

        clips = torch.stack(clips).permute(1, 0, 2, 3, 4)  # B, T, C, H, W -> T, B, C, H, W.
        targets = torch.from_numpy(np.array(targets))

        return FixedLengthBatch(clips, targets, filenames)

    def collate_poses_variable_length(batch):
        clips = [e[0] for e in batch]
        targets = [e[1] for e in batch]
        lengths = [len(c) for c in clips]
        filenames = [e[2] for e in batch]

        clips = torch.nn.utils.rnn.pad_sequence(clips, batch_first=False)
        lengths = torch.from_numpy(np.array(lengths)).long()
        targets = torch.from_numpy(np.array(targets))

        return VariableLengthBatch(clips, lengths, targets, filenames)

    def collate_poses_inference(batch):
        clips = [e[0] for e in batch]
        lengths = [len(c) for c in clips]
        filenames = [e[1] for e in batch]

        clips = torch.nn.utils.rnn.pad_sequence(clips, batch_first=False)

        return InferenceBatch(clips, lengths, filenames)

    def collate_poses(batch):
        clips = [e[0] for e in batch]
        targets = [e[1] for e in batch]
        filenames = [e[2] for e in batch]

        clips = torch.stack(clips).permute(1, 0, 2)  # B, T, C -> T, B, C.
        targets = torch.from_numpy(np.array(targets))

        return FixedLengthBatch(clips, targets, filenames)

    if 'Mediapipe' in data_kind or 'OpenPose' in data_kind or 'MMPose' in data_kind:
        if inference_mode:
            return collate_poses_inference
        elif variable_length:
            return collate_poses_variable_length
        else:
            return collate_poses
    elif data_kind == 'RGB_Full':
        return collate_images
    else:
        raise ValueError(f'Unknown dataset kind {data_kind}.')
