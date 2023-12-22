"""Dataset that loads pre-extracted pose keypoints."""
import glob
import os
from typing import List, Dict

import cv2
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset

from . import custom_transforms as CT
from .common import collect_samples, Sample
from .. import data_preprocessing
from ..data_preprocessing.mediapipe_keypoints import extract


def get_transforms(hparams: Dict, *, train: bool):
    if hparams['data_kind'].startswith('OpenPose'):
        keypoints_to_drop = np.array([8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24])  # Lower body.
        if train:
            return T.Compose([
                CT.ChangeSpeed(hparams['augment_adaptSpeed'], hparams['augment_adaptSpeedFaster']) if hparams[
                                                                                                          'augment_adaptSpeed'] > 0.0 else CT.Passthrough(),
                CT.TemporalShift(hparams['augment_temporalShift']) if hparams[
                                                                          'augment_temporalShift'] > 0.0 else CT.Passthrough(),
                CT.RandomHorizontalFlip(0.5) if hparams['augment_hflip'] else CT.Passthrough(),
                CT.DropKeypoints(keypoints_to_drop),
                CT.Force2D(),  # Drop confidence value.
                CT.Ravel(),
                CT.KeypointsToTensor()
            ])
        else:
            return T.Compose([
                CT.DropKeypoints(keypoints_to_drop),
                CT.Force2D(),
                CT.Ravel(),
                CT.KeypointsToTensor()
            ])
    elif hparams['data_kind'].startswith('MMPose'):
        keypoints_to_drop = np.arange(11, 91)  # Lower body, feet, face. See https://github.com/jin-s13/COCO-WholeBody.
        if train:
            return T.Compose([
                # TODO HARDCODED FOR VGT
                CT.Normalize(960, 540) if 'Raw' in hparams['data_kind'] else CT.Passthrough(),
                CT.ChangeSpeed(hparams['augment_adaptSpeed'], hparams['augment_adaptSpeedFaster']) if hparams[
                                                                                                          'augment_adaptSpeed'] > 0.0 else CT.Passthrough(),
                CT.TemporalShift(hparams['augment_temporalShift']) if hparams[
                                                                          'augment_temporalShift'] > 0.0 else CT.Passthrough(),
                CT.RandomHorizontalFlip(0.5) if hparams['augment_hflip'] else CT.Passthrough(),
                CT.DropKeypoints(keypoints_to_drop),
                CT.Force2D(),  # Drop confidence value.
                CT.Ravel(),
                CT.KeypointsToTensor()
            ])
        else:
            return T.Compose([
                CT.DropKeypoints(keypoints_to_drop),
                CT.Force2D(),
                CT.Ravel(),
                CT.KeypointsToTensor()
            ])
    elif hparams['data_kind'].startswith('Mediapipe'):
        keypoints_to_drop = np.arange(25, 33)  # Lower body.
        if train:
            return T.Compose([
                CT.ChangeSpeed(hparams['augment_adaptSpeed'], hparams['augment_adaptSpeedFaster']) if hparams[
                                                                                                          'augment_adaptSpeed'] > 0.0 else CT.Passthrough(),
                CT.TemporalShift(hparams['augment_temporalShift']) if hparams[
                                                                          'augment_temporalShift'] > 0.0 else CT.Passthrough(),
                CT.RandomYAxisRotation(hparams['augment_rotY']) if hparams['augment_rotY'] > 0 else CT.Passthrough(),
                CT.RandomHorizontalFlip(0.5) if hparams['augment_hflip'] else CT.Passthrough(),
                CT.DropKeypoints(keypoints_to_drop),
                CT.ReplaceNaN(),
                CT.Force2D() if hparams['mediapipe_2d'] else CT.Passthrough(),
                CT.Ravel(),
                CT.KeypointsToTensor()
            ])
        else:
            return T.Compose([
                CT.DropKeypoints(keypoints_to_drop),
                CT.ReplaceNaN(),
                CT.Force2D() if hparams['mediapipe_2d'] else CT.Passthrough(),
                CT.Ravel(),
                CT.KeypointsToTensor()
            ])
    else:
        raise ValueError('No transforms implemented for data kind {}'.format(hparams['data_kind']))


class PoseInferenceDataset(Dataset):
    def __init__(self, directory: str, data_kind: str, file_extension: str, hparams: Dict):
        super(PoseInferenceDataset, self).__init__()

        if file_extension[0] == '.':
            file_extension = file_extension[1:]
        self.file_extension = file_extension

        print(f'Inference dataset will load files from {directory}.')

        self.videos = True
        if 'jpg' in file_extension or 'png' in file_extension:
            print(f'File extension is an image extension ({file_extension}). Will load image directories.')
            self.videos = False

        self.path = directory
        self.data_kind = data_kind
        if self.videos:
            self.files = sorted(glob.glob(os.path.join(self.path, f'*.{file_extension}')))  # Video files.
        else:
            self.files = sorted(glob.glob(os.path.join(self.path, '*/')))  # Image directories.
        self.transforms = get_transforms(hparams, train=False)
        print(self.transforms, flush=True)
        self.sharpen_flag = hparams.get("sharpen", False)
        self.sharpen_sigma = hparams.get("sharpen_sigma", 0)
        self.debug = hparams.get("debug", False)

    def __getitem__(self, item):
        """Get the input features and filename as a tuple."""
        frames = []
        if self.videos:
            # Load video.
            cap = cv2.VideoCapture(self.files[item])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            # Load image directory.
            for file in sorted(glob.glob(os.path.join(self.files[item], f'*.{self.file_extension}'))):
                frame = cv2.imread(file)
                if frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        height, width, _ = frames[0].shape

        # Perform any processing on the video, like extracting keypoints.
        if self.data_kind.startswith('Mediapipe'):
            keypoints = self._extract_keypoints(np.stack(frames), self.sharpen_flag, self.sharpen_sigma)
            if self.data_kind.startswith('Mediapipe_Cleaned'):
                keypoints = self._process_keypoints(keypoints, height / width)

            if self.debug:
                mp_failures = self._count_nans(keypoints)
                return keypoints, self.files[item], mp_failures
            keypoints = np.nan_to_num(keypoints)  # Clean up any remaining NaNs.
            keypoints = self.transforms(keypoints)
            return keypoints, self.files[item]
        else:
            raise ValueError(f'Unsupported data kind {self.data_kind}')

    def _extract_keypoints(self, frames: np.ndarray, sharpen: bool, sharpen_sigma: float) -> np.ndarray:
        if sharpen:
            return extract(frames, sharpen_fn=sharpen, sharpen_sigma=sharpen_sigma)
        else:
            return extract(frames)

    def _process_keypoints(self, keypoints: np.ndarray, inverse_aspect_ratio: float) -> np.ndarray:
        return data_preprocessing.mediapipe_keypoints.postprocess(keypoints, inverse_aspect_ratio)

    def _count_nans(self, keypoints):
        # Gets number of frames with at least one missing keypoint.
        keypoints = keypoints.reshape(-1, keypoints.shape[1] * keypoints.shape[2])
        return np.sum(np.sum(np.isnan(keypoints), axis=1) > 0)

    def __len__(self):
        return len(self.files)


class RawPoseDataset(Dataset):
    def __init__(self, job: str, pose_format: str, **kwargs):
        super(RawPoseDataset, self).__init__()

        self.root_path = kwargs['data_dir']
        self.job = job
        self.samples_file_override = kwargs['samples_file_override']
        self.variable_length = kwargs['variable_length_sequences']
        self.pose_format = pose_format

        self.transform = get_transforms(dict(**kwargs), train=self.job == 'train')
        print(self.transform, flush=True)

        self.samples = self._collect_samples()

    def __getitem__(self, item):
        sample = self.samples[item]
        sample_filename, _ = os.path.splitext(sample.path)
        frames = np.load(os.path.join(self.root_path, self.pose_format, sample_filename + '.npy'))
        if sample.frame_indices is None:
            clip = frames[self._get_frame_indices(len(frames), self.variable_length)]
        else:
            clip = frames[sample.frame_indices]

        clip = self.transform(clip)

        return clip, sample.label, sample.path

    def _get_frame_indices(self, num_frames: int, variable_length: bool) -> List[int]:
        if variable_length:
            return list(range(0, num_frames, 1))
        # TODO: Make these values configurable.
        sequence_length = 16
        temporal_stride = 2
        frame_start = (num_frames - sequence_length) // (2 * temporal_stride)
        frame_end = frame_start + sequence_length * temporal_stride
        if frame_start < 0:
            frame_start = 0
        if frame_end > num_frames:
            frame_end = num_frames
        frame_indices = list(range(frame_start, frame_end, temporal_stride))
        while len(frame_indices) < sequence_length:
            # Pad
            frame_indices.append(frame_indices[-1])
        return frame_indices

    def __len__(self):
        return len(self.samples)

    def _collect_samples(self) -> List[Sample]:
        return collect_samples(self.root_path, self.job, self.samples_file_override)


class CleanedPoseDataset(Dataset):
    def __init__(self, job: str, pose_format: str, **kwargs):
        super(CleanedPoseDataset, self).__init__()

        self.pose_format = pose_format
        self.root_path = kwargs['data_dir']
        self.job = job
        self.samples_file_override = kwargs['samples_file_override']
        self.variable_length = kwargs['variable_length_sequences']

        self.transform = get_transforms(dict(**kwargs), train=self.job == 'train')
        print(self.transform, flush=True)

        self.samples = self._collect_samples()

    def __getitem__(self, item):
        sample = self.samples[item]
        sample_filename, _ = os.path.splitext(sample.path)
        frames = np.load(os.path.join(self.root_path, self.pose_format, 'joints', sample_filename + '.npy'))

        if sample.frame_indices is None:
            clip = frames[self._get_frame_indices(len(frames), self.variable_length)]
        else:
            clip = frames[sample.frame_indices]

        clip = self.transform(clip)

        return clip, sample.label, sample.path

    def _get_frame_indices(self, num_frames: int, variable_length: bool) -> List[int]:
        if variable_length:
            return list(range(0, num_frames, 1))
        # TODO: Make these values configurable.
        sequence_length = 16
        temporal_stride = 2
        frame_start = (num_frames - sequence_length) // (2 * temporal_stride)
        frame_end = frame_start + sequence_length * temporal_stride
        if frame_start < 0:
            frame_start = 0
        if frame_end > num_frames:
            frame_end = num_frames
        frame_indices = list(range(frame_start, frame_end, temporal_stride))
        while len(frame_indices) < sequence_length:
            # Pad
            frame_indices.append(frame_indices[-1])
        return frame_indices

    def __len__(self):
        return len(self.samples)

    def _collect_samples(self) -> List[Sample]:
        return collect_samples(self.root_path, self.job, self.samples_file_override)
