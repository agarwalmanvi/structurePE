import numpy as np
import torch


# # # # Grooving similarity # # # #

def get_gd(arr, task):

    b, t, d = arr.shape

    if task == "accompaniment_generation":

        pre = np.concatenate([
            np.zeros((b, 2, d)), arr[:, :-1, :]
        ], axis=1)
        post = np.concatenate([
            np.zeros((b, 1, d)), arr
        ], axis=1)
        soft_onsets = (pre - post)[:, 1:, :]
        soft_onsets[soft_onsets > 0] = 0.0
        soft_onsets = np.abs(soft_onsets)

        soft_onset_count = np.sum(soft_onsets, axis=-1)
        soft_onset_count = torch.tensor(soft_onset_count).view(b, -1, 4).sum(dim=-1)


    else:

        arr_tracks = np.reshape(arr, (b, t, 3, 128))

        pre = np.concatenate([
            np.zeros((b, 2, 3, 128)), arr_tracks[:, :-1, :, :]
        ], axis=1)
        post = np.concatenate([
            np.zeros((b, 1, 3, 128)), arr_tracks
        ], axis=1)
        soft_onsets = (pre - post)[:, 1:, :, :]
        soft_onsets[soft_onsets > 0] = 0.0
        soft_onsets = np.abs(soft_onsets)

        soft_onset_count = np.sum(soft_onsets, axis=-1)
        # b x t x 3 -> b x n_16th_beats x 4 x 3 -> b x n_16th_beats -> b x n_bars x 16
        soft_onset_count = torch.tensor(soft_onset_count).view(b, -1, 4, 3).sum(dim=(-1, -2))

    return soft_onset_count


def onset_metric(pred, tgt, task):
    grooving_pattern_per_sample_pred = get_gd(pred, task)
    pred_norm = torch.nn.functional.normalize(grooving_pattern_per_sample_pred, dim=-1).numpy()
    grooving_pattern_per_sample_tgt = get_gd(tgt, task)
    tgt_norm = torch.nn.functional.normalize(grooving_pattern_per_sample_tgt, dim=-1).numpy()

    bins = 20
    density = False
    mean_diff_per_sample = []

    for bix in range(tgt_norm.shape[0]):
        one_sample_pred = pred_norm[bix, :]

        one_sample_tgt = tgt_norm[bix, :]

        range_ = (
            np.min([
                np.min(one_sample_pred), np.min(one_sample_tgt)
            ]),
            np.max([
                np.max(one_sample_pred), np.max(one_sample_tgt)
            ])
        )

        counts_per_bin_pred, _ = np.histogram(one_sample_pred, bins=bins, range=range_, density=density)
        counts_per_bin_tgt, _ = np.histogram(one_sample_tgt, bins=bins, range=range_, density=density)

        histogram_overlap = np.min(np.stack((counts_per_bin_pred, counts_per_bin_tgt), axis=0), axis=0)

        mean_diff_per_sample.append(np.mean(histogram_overlap))

    return np.array(mean_diff_per_sample)


# # # # Chroma Similarity # # # #

def get_chroma_onsets(arr, task):
    b, t, d = arr.shape

    if task == "accompaniment_generation":

        # get onsets without collapsing last dimension
        pre = np.concatenate([
            np.zeros((b, 2, d)), arr[:, :-1, :]
        ], axis=1)
        post = np.concatenate([
            np.zeros((b, 1, d)), arr
        ], axis=1)
        # b x t x d
        soft_onsets = (pre - post)[:, 1:, :]
        soft_onsets[soft_onsets > 0] = 0.0
        soft_onsets = np.abs(soft_onsets)

        # get counts of onsets for each chroma
        soft_onsets = torch.tensor(soft_onsets)
        extra_onsets = torch.zeros((b, t, 4))
        soft_onsets = torch.concat((soft_onsets, extra_onsets), dim=-1)
        # b x t x 12
        n_onsets_per_chroma = torch.sum(soft_onsets.view(b, t, -1, 12), dim=2)

        d_unit = 1 * 2 * 16
        # b x n_units x 12
        n_onsets_per_chroma_per_unit = torch.sum(
            n_onsets_per_chroma.view(b, -1, d_unit, 12), dim=2
        )

    else:

        arr_tracks = np.reshape(arr, (b, t, 3, 128))
        # get onsets without collapsing any dimension
        pre = np.concatenate([
            np.zeros((b, 2, 3, 128)), arr_tracks[:, :-1, :, :]
        ], axis=1)
        post = np.concatenate([
            np.zeros((b, 1, 3, 128)), arr_tracks
        ], axis=1)
        # b x t x d
        soft_onsets = (pre - post)[:, 1:, :, :]
        soft_onsets[soft_onsets > 0] = 0.0
        soft_onsets = np.abs(soft_onsets)

        # get counts of onsets for each chroma
        soft_onsets = torch.tensor(soft_onsets)
        extra_onsets = torch.zeros((b, t, 3, 4))
        soft_onsets = torch.concat((soft_onsets, extra_onsets), dim=-1)
        # b x t x 3 x 12
        n_onsets_per_chroma = torch.sum(soft_onsets.view(b, t, 3, -1, 12), dim=3)

        d_unit = 1 * 2 * 16
        # b x n_units x 3 x 12
        n_onsets_per_chroma_per_unit = torch.sum(
            n_onsets_per_chroma.view(b, -1, d_unit, 3, 12), dim=2
        )
        n_onsets_per_chroma_per_unit = n_onsets_per_chroma_per_unit.view(b, -1, 3 * 12)

    return n_onsets_per_chroma_per_unit


def get_chroma_similarity(pred, tgt, task):
    n_onsets_pred = get_chroma_onsets(pred, task)
    n_onsets_tgt = get_chroma_onsets(tgt, task)

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    # b x n_units
    scores = cos(n_onsets_pred, n_onsets_tgt)

    # b
    score_per_sample = torch.mean(scores, dim=-1).numpy() * 100

    return score_per_sample

# # # # Grooving similarity # # # #

def get_chroma_counts_per_half_beat(arr, task):
    b, t, d = arr.shape

    # get counts for each chroma
    t_arr = torch.tensor(arr)

    if task == "accompaniment_generation":

        extra_pitches = torch.zeros((b, t, 4))
        t_arr = torch.concat((t_arr, extra_pitches), dim=-1)
        # b x t x 12
        counts_per_chroma = torch.sum(t_arr.view(b, t, -1, 12), dim=2)

    else:

        t_arr = t_arr.view(b, t, 3, 128)
        extra_pitches = torch.zeros((b, t, 3, 4))
        t_arr = torch.concat((t_arr, extra_pitches), dim=-1)
        # b x t x 12
        counts_per_chroma = torch.sum(t_arr.view(b, t, 3, -1, 12), dim=(2, 3))

    d_unit = int(16 / 8)
    # b x n_half_beats x 12
    counts_per_chroma_per_half_beat = torch.sum(
        counts_per_chroma.view(b, -1, d_unit, 12), dim=2
    )

    return counts_per_chroma_per_half_beat

# # # # Self similarity matrix distance # # # #

def get_ssm(arr, task):
    counts_arr = get_chroma_counts_per_half_beat(arr, task)
    ssm = torch.cosine_similarity(
        counts_arr.unsqueeze(2), counts_arr.unsqueeze(1), dim=-1
    )
    print(ssm.shape)
    return ssm


def get_ssm_distance(pred, tgt, task):
    ssm_distances = []
    ssms_pred = []
    ssms_tgt = []

    sub_len = 64
    if tgt.shape[0] > sub_len:

        print("Breaking up the sample, batch size is too big: ", tgt.shape[0])
        idx_starts = np.arange(start=0, stop=tgt.shape[0], step=sub_len)

        for idx_start in idx_starts:
            print("IDX start: ", idx_start)

            pred_sub = pred[idx_start:idx_start + sub_len, :, :]
            tgt_sub = tgt[idx_start:idx_start + sub_len, :, :]

            ssm_pred_sub = get_ssm(pred_sub, task)
            ssms_pred.append(ssm_pred_sub)
            ssm_tgt_sub = get_ssm(tgt_sub, task)
            ssms_tgt.append(ssm_tgt_sub)

            metric_sub = torch.mean(torch.abs(ssm_pred_sub - ssm_tgt_sub), dim=(-1, -2)) * 100
            ssm_distances.append(metric_sub)

    ssm_distance = torch.cat(ssm_distances, dim=0).numpy()
    ssm_pred = torch.cat(ssms_pred, dim=0).numpy()
    ssm_tgt = torch.cat(ssms_tgt, dim=0).numpy()

    return ssm_distance, (ssm_pred, ssm_tgt)

# # # # Note Density Distance # # # #

def get_note_density(arr):
    n_voices_per_timestep = np.sum(arr, axis=-1)
    b, t = n_voices_per_timestep.shape
    n_voices_per_timestep = np.reshape(n_voices_per_timestep, (b, -1, 4))
    n_voices_per_16th_beat = np.sum(n_voices_per_timestep, -1)
    return n_voices_per_16th_beat


def get_nd(pred, tgt):
    nd_pred = get_note_density(pred)
    nd_tgt = get_note_density(tgt)

    nd = np.mean(
        np.abs(nd_pred - nd_tgt),
        axis=-1
    )
    return nd