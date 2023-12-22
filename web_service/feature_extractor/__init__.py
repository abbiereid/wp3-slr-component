import numpy as np
from .slr_pipeline.src.slr.models.module import Module
from .slr_pipeline.src.slr import data_preprocessing
from .slr_pipeline.src.slr.data.pose import get_transforms
import torch
import cv2


def load_model(checkpoint_path):
    print(f'Loading module from checkpoint {checkpoint_path}')
    module = Module.load_from_checkpoint(checkpoint_path)
    print('Model loaded')
    module = module.eval()
    if torch.cuda.is_available():
        print('Using CUDA')
        module = module.cuda()
    else:
        print('No CUDA available')
    return module


class Batch:
    def __init__(self, inputs: torch.Tensor, lengths):
        self.inputs = inputs
        self.lengths = lengths

    def batch_size(self):
        return self.inputs.size(1)  # batch_first = False.

    def to(self, device):
        self.inputs = self.inputs.to(device)
        return self


def extract_features_blocking(filename, sourceLanguage, module):
    print(f'Extracting features from {filename} for language {sourceLanguage}')

    hook_outputs = []

    def _hook(self, input, output):
        hook_outputs.append(output)

    module.model.setup_inference_hook('spatial', _hook)

    # Load video.
    frames = []
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    height, width, _ = frames[0].shape

    # Perform any processing on the video, like extracting keypoints.
    keypoints = data_preprocessing.mediapipe_keypoints.extract(np.stack(frames))
    keypoints = data_preprocessing.mediapipe_keypoints.postprocess(keypoints, height / width)

    keypoints = np.nan_to_num(keypoints)  # Clean up any remaining NaNs.

    transforms = get_transforms(module.hparams, train=False)
    keypoints = transforms(keypoints)

    batch = Batch(keypoints, torch.from_numpy(np.array([len(keypoints)])))

    # --- Inference --- #
    with torch.no_grad():
        if torch.cuda.is_available():
            batch = batch.to('cuda')

        batch.inputs = batch.inputs.unsqueeze(1)  # batch_first=False.

        _model_outputs = module(batch)

        output = hook_outputs[0][:, 0]  # Get from hook storage.
        output_array = output.detach().cpu().numpy()

        module.model.reset_inference_hook('spatial')

        return output_array
