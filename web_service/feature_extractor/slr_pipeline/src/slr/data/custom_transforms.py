"""Custom data transforms, e.g., for pose data."""
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class Normalize:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        poses[..., 0] /= self.w
        poses[..., 1] /= self.h
        return poses


class Force2D:
    def __init__(self):
        super(Force2D, self).__init__()

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        assert poses.shape[-1] == 3
        return poses[..., :2]


class Ravel:
    def __init__(self):
        super(Ravel, self).__init__()

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        return np.reshape(poses, (poses.shape[0], poses.shape[1] * poses.shape[2]))


class Passthrough:
    def __init__(self):
        pass

    def __call__(self, poses):
        return poses


class TemporalShift:
    def __init__(self, p_apply=0.5, max_offset=5):
        self.p_apply = p_apply
        self.max_offset = max_offset

    def __call__(self, poses):
        if len(poses) < self.max_offset * 3 or random.random() >= self.p_apply:  # Don't apply on very short sequences.
            return poses
        new_start = random.randint(1, 1 + self.max_offset)
        new_end = random.randint(len(poses) - self.max_offset, len(poses) - 1)
        return poses[new_start:new_end]


class ChangeSpeed:
    def __init__(self, p_apply=0.5, p_faster=0.5):
        self.p_apply = p_apply
        self.p_faster = p_faster

    def __call__(self, poses):
        if random.random() < self.p_apply:
            faster = random.random() < self.p_faster
            if faster:
                # Drop frames.
                poses = poses[::2]
                return poses
            else:
                # Interpolate to add frames.
                interpolated_poses = np.zeros((poses.shape[0] * 2 - 1, *poses.shape[1:]))
                for i in range(0, poses.shape[0]):
                    interpolated_poses[i * 2] = poses[i]
                for i in range(1, interpolated_poses.shape[0] - 1, 2):
                    interpolated_poses[i] = 0.5 * (interpolated_poses[i - 1] + interpolated_poses[i + 1])
                return interpolated_poses
        return poses


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()

        self.p = p

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            poses[..., 0] *= -1  # Flip the x coordinate.
        return poses


class RandomYAxisRotation:
    def __init__(self, max_degrees: int):
        super(RandomYAxisRotation, self).__init__()

        self.max_degrees = max_degrees

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        angle = random.randint(-self.max_degrees, self.max_degrees)
        R = Rotation.from_euler('y', angle, degrees=True)
        for i in range(len(poses)):
            poses[i] = (R.as_matrix() @ poses[i].T).T
        return poses


class KeypointsToTensor:
    def __init__(self):
        super(KeypointsToTensor, self).__init__()

    def __call__(self, poses: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(poses).float()


class DropKeypoints:
    """Drop lower body keypoints.
    """

    def __init__(self, keypoints_to_drop):
        super(DropKeypoints, self).__init__()
        # 25 - 33: lower body.
        self.keypoints_to_drop = keypoints_to_drop

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        """Performs the transformation.

        :param poses: An array of keypoints of shape (time, keypoints, coordinates).
        :return: An array of keypoints of shape (time, new_keypoints, coordinates).
        """
        transformed = np.delete(poses, self.keypoints_to_drop, axis=1)
        return transformed

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'DropKeypoints()'


class ReplaceNaN:
    """Replace NaN with 0."""

    def __init__(self):
        pass

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        return np.nan_to_num(poses)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'ReplaceNaN()'
