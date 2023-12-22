import argparse
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch

import pandas  as pd
import matplotlib.pyplot as plt

from slr.data.module import get_inference_data_loader
from slr.models.module import Module
from train import _get_git_commit_hash


def predict(args):
    # --- Initialization --- #
    # module = Module.load_from_checkpoint(args.checkpoint)
    # hparams = dict(module.hparams)
    hparams = dict()

    # pl.seed_everything(hparams['seed'])

    # Override the data directory.
    hparams['data_dir'] = args.data
    # # Override the batch size.
    hparams['batch_size'] = 1
    # Override the number of workers
    hparams['num_workers'] = args.num_workers
    hparams["debug"] = True

    tests = [
        {
            "sharpen": False,
            "sharpen_sigma": None
        },
        {
            "sharpen": True,
            "sharpen_sigma": None
        },
        {
            "sharpen": True,
            "sharpen_sigma": 1
        },
        {
            "sharpen": True,
            "sharpen_sigma": 2
        }

    ]
    for test_i in range(len(tests)):
        hparams["sharpen"] = tests[test_i]["sharpen"]
        hparams["sharpen_sigma"] =  tests[test_i]["sharpen_sigma"]
        data_loader = get_inference_data_loader(**hparams, data_kind="Mediapipe")
        file_scores = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                model_inputs, filenames, mp_failures = batch
                mp_failures = mp_failures.cpu().numpy().item()
                num_frames = model_inputs.shape[1]
                percentage_frames_failures =(mp_failures/num_frames)*100
                filename = "".join(filenames)
                file_scores.append((filename, percentage_frames_failures))
                print("MediaPipe Report:\nFile name: {}\nShapen: {}\nSharpen signma: {}\nPercentage frame failures: {}%".format(
                    filename,
                    hparams["sharpen"],
                    hparams["sharpen_sigma"], 
                    round(percentage_frames_failures,2)
                    )
                )
                
                tests[test_i]["file_mp_failures"] = file_scores
    testing_logs_dir = os.path.join("testing_logs", "mediapipe")
    if not os.path.exists(testing_logs_dir): os.makedirs(testing_logs_dir)
    testing_file_path = os.path.join(testing_logs_dir, "mediapipe_testing.json")
    with open(testing_file_path, "w") as f:
        json.dump(tests, f)

    plt.figure(figsize=(20,10))   
    for log in tests:
        log_df = pd.DataFrame(log["file_mp_failures"], columns=["video", "percentages"])
        sharpen = "sharpen" if log["sharpen"] else ""
        sharpen_sigma = "sharpen_sigma" if log["sharpen_sigma"] else ""
        plt.plot(log_df.percentages, label = "sharpen {}, sigma {}".format(log["sharpen"], log["sharpen_sigma"]))
     
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(testing_logs_dir, "mediapipe_test.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='The path to the directory containing the videos.')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing.', default=1)
    parser.add_argument('--num_workers', type=int, help='Number of data loader workers.', default=1)

    args = parser.parse_args()

    predict(args)
