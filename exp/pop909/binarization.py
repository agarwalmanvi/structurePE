# get_prob, threshold, get_tgt, get_mse
import numpy as np
from pathlib import Path

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_prob(f):
    logits = f['logits']
    probabilities = sigmoid(logits)
    return probabilities

def use_threshold_merge(thr, s, pred_sample):
    # sample.shape = b x t x d
    pred_sample = use_threshold(thr, pred_sample)
    return_sample = np.zeros_like(pred_sample)

    for b_idx in range(pred_sample.shape[0]):
        for d_idx in range(pred_sample.shape[-1]):

            sample = pred_sample[b_idx, :, d_idx]
            idx = np.nonzero(sample)[0]
            n_empty_cells = np.diff(idx) - 1

            for dist_idx, dist in enumerate(n_empty_cells):
                if dist <= s:
                    idx_pre = idx[dist_idx]
                    idx_post = idx[dist_idx + 1]
                    sample[idx_pre:idx_post] = 1.0

            return_sample[b_idx, :, d_idx] = sample

    return return_sample

def use_threshold(thr, arr):
    return (arr >= thr).astype(int)


def get_tgt(fpaths, task):
    xs = []
    ys = []

    for fpath in fpaths:
        fpath_parts = fpath.split("/")
        corrected_path = Path(
            "/path/to/structurePE/data/pop909/data"
        ) / fpath_parts[-3] / fpath_parts[-2] / fpath_parts[-1]

        data = np.load(corrected_path)['data']
        if task == "accompaniment_generation":
            x = data[:256, :-1]
            y = data[256:, :-1]
        elif task == "next_note_prediction":
            x = data[:, :-1]
            y = data[:, 1:]
        xs.append(x)
        ys.append(y)

    X = np.stack(xs, axis=0)
    # give size b x t x d
    X = np.transpose(X, (0, 2, 1))
    Y = np.stack(ys, axis=0)
    Y = np.transpose(Y, (0, 2, 1))

    return X, Y

def get_mse(tgt, pred):
    return np.mean((tgt - pred)**2)