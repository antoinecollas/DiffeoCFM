from itertools import product
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.covariance import OAS
import time

from cov_est import cov_est
from data import load_data
from fm import DiffeoCFM
from constants import (
    EXPE,
    N_JOBS,
    N_SPLITS,
    TEST_SIZE,
    DATASETS,
    METHODS,
    ATLAS,
    NORMALIZE,
    PATH_RESULTS,
)


def run_split(
    split,
    cov_train,
    cov_val,
    y_train,
    y_val,
    groups_train,
    groups_val,
    model,
    path_results,
):
    # Assert subject IDs are disjoint
    assert set(groups_train).isdisjoint(set(groups_val)), (
        "Groups are not disjoint between train and val sets."
    )

    # Training
    training_start = time.time()

    train_info = model.fit(cov_train, y_train)
    if train_info is not None:
        train_losses_epoch = train_info["train_loss"]
        val_losses_epoch = train_info["val_loss"]

    training_time = time.time() - training_start

    # Sampling
    sampling_start = time.time()

    sol_train = model.sample(y_train)
    sol_val = model.sample(y_val)

    sampling_time = time.time() - sampling_start

    def path_maker(end_path):
        return path_results / f"split_{split}_{end_path}.npy"

    if isinstance(model, DiffeoCFM):
        np.save(path_maker("train_losses"), train_losses_epoch)
        np.save(path_maker("val_losses"), val_losses_epoch)

    np.save(path_maker("covariances_train"), cov_train)
    np.save(path_maker("conditionals_train"), y_train)
    np.save(path_maker("groups_train"), groups_train)

    np.save(path_maker("covariances_val"), cov_val)
    np.save(path_maker("conditionals_val"), y_val)
    np.save(path_maker("groups_val"), groups_val)

    np.save(path_maker("covariances_generated_samples_train"), sol_train)
    np.save(path_maker("conditionals_generated_samples_train"), y_train)
    np.save(path_maker("covariances_generated_samples_val"), sol_val)
    np.save(path_maker("conditionals_generated_samples_val"), y_val)

    np.save(path_maker("training_time"), np.array([training_time]))
    np.save(path_maker("sampling_time"), np.array([sampling_time]))


def generate_splits_with_left_out_group(
    X, y, groups, left_out_group, n_splits=10, rng=None
):
    train_idx = np.where(groups != left_out_group)[0]
    test_idx = np.where(groups == left_out_group)[0]

    subsplits = list()
    if rng is None:
        rng = np.random.RandomState(0)

    kf_train = KFold(n_splits=n_splits, shuffle=True, random_state=rng)
    kf_test = KFold(n_splits=n_splits, shuffle=True, random_state=rng)
    for (new_train, _), (new_test, _) in zip(
        kf_train.split(train_idx), kf_test.split(test_idx)
    ):
        subsplits.append((train_idx[new_train], test_idx[new_test]))

    return subsplits


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    for i, (dataset, method) in enumerate(product(DATASETS, METHODS)):
        diffeo_name = method["diffeo"]
        model = method["model"]

        model_name = model.__class__.__name__

        print(f"Loading data for {dataset}...")
        ts, y, groups = load_data(dataset, ATLAS, rng)

        print("Estimating covariances...")
        cov = cov_est(ts, n_jobs=N_JOBS, normalize=NORMALIZE)

        if EXPE == "eeg":
            # filter covariance matrices with absolute values above 1e4
            mask_abs = np.max(np.abs(cov), axis=(-2, -1)) < 1e4

            # filter covariance matrices with Mahalanobis distance
            oas = OAS()
            oas.fit(cov.reshape(cov.shape[0], -1))
            cov_inv = np.linalg.inv(oas.covariance_)
            mahalanobis_distances = np.array(
                [
                    mahalanobis(
                        cov[i].flatten(), np.mean(cov, axis=0).flatten(), cov_inv
                    )
                    for i in range(cov.shape[0])
                ]
            )
            mask_maha = mahalanobis_distances < np.percentile(mahalanobis_distances, 90)

            # combine masks
            mask = mask_abs & mask_maha
            cov, y, groups = cov[mask], y[mask], groups[mask]

        if EXPE == "fmri":
            left_out_groups = [None]
        else:
            assert EXPE == "eeg", f"Unknown experiment {EXPE}"
            left_out_groups = np.arange(len(np.unique(groups)))

        for group in left_out_groups:
            if group is not None:
                print(
                    f"Training and generating samples for {dataset} using {model_name} with {diffeo_name} diffeomorphism, left out session {group}..."
                )
            else:
                print(
                    f"Training and generating samples for {dataset} using {model_name} with {diffeo_name} diffeomorphism..."
                )

            PATH_RESULTS_DATASET_GROUP_METHOD = PATH_RESULTS / f"{dataset}_{ATLAS}"
            PATH_RESULTS_DATASET_GROUP_METHOD = (
                PATH_RESULTS_DATASET_GROUP_METHOD
                / f"group_{group}"
                / f"{diffeo_name}_{model_name}"
            )
            PATH_RESULTS_DATASET_GROUP_METHOD.mkdir(parents=True, exist_ok=True)

            if EXPE == "fmri":
                ss = GroupShuffleSplit(
                    n_splits=N_SPLITS, random_state=rng, test_size=TEST_SIZE
                )
                splits = list(ss.split(cov, y, groups=groups))
            else:
                splits = generate_splits_with_left_out_group(
                    cov, y, groups, left_out_group=group, n_splits=N_SPLITS, rng=rng
                )

            Parallel(n_jobs=N_JOBS)(
                delayed(run_split)(
                    split=split,
                    cov_train=cov[train_idx],
                    cov_val=cov[val_idx],
                    y_train=y[train_idx],
                    y_val=y[val_idx],
                    groups_train=groups[train_idx],
                    groups_val=groups[val_idx],
                    model=model,
                    path_results=PATH_RESULTS_DATASET_GROUP_METHOD,
                )
                for split, (train_idx, val_idx) in enumerate(splits)
            )

            print(f"Finished processing {dataset} using {model_name}.")
            print()
