from joblib import Parallel, delayed
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from EvaGeM.distribution_based import alpha_precision, beta_recall
from prdc import compute_prdc

from constants import (
    N_JOBS,
    DATASETS,
    EXPE,
    PATH_RESULTS,
    PATH_FIGURES,
    NORMALIZE,
)


def gather_results_structure(base_path: Path, datasets: list[str]):
    """
    Walks ``base_path`` looking for
      results/
        <dataset>/
          <group>/
            <method>/
              split_*.npy
    and builds a nested dict:
      {dataset: {group: {method: n_splits}}}
    """
    results: dict[str, dict[str, dict[str, int]]] = {}

    for dataset_path in base_path.iterdir():
        dataset = dataset_path.name

        # dataset is dataset_name + "_" + atlas ("msdl", "schaeffer_100", or "None")
        # remove the atlas part
        while dataset and "_" in dataset and dataset not in datasets:
            dataset = "_".join(dataset.split("_")[:-1])

        if not dataset_path.is_dir() or dataset not in datasets:
            continue
        ds_name = dataset_path.name
        results[ds_name] = {}

        for group_path in dataset_path.iterdir():
            if not group_path.is_dir():
                continue
            grp = group_path.name
            results[ds_name][grp] = {}

            for method_path in group_path.iterdir():
                if not method_path.is_dir():
                    continue
                mth = method_path.name

                # collect unique split indices
                split_ids = set()
                for file in method_path.glob("split_*"):
                    match = re.match(r"split_(\d+)_", file.name)
                    if match:
                        split_ids.add(int(match.group(1)))

                results[ds_name][grp][mth] = len(split_ids)

    return results


def fraction_covariance_matrix(matrices, tol=1e-12):
    is_sym = np.empty(matrices.shape[0], dtype=bool)
    is_pos_def = np.empty(matrices.shape[0], dtype=bool)
    is_cov = np.empty(matrices.shape[0], dtype=bool)

    for i, mat in enumerate(matrices):
        # Symmetric
        is_sym[i] = np.allclose(mat, mat.T, atol=tol)

        # Positive definite
        eigvals = np.linalg.eigvalsh(mat)
        is_pos_def[i] = np.all(eigvals > tol)

        # correlation matrix
        is_cov[i] = is_sym[i] and is_pos_def[i]

    return {
        "Sym.": np.mean(is_sym),
        "Pos. def.": np.mean(is_pos_def),
        "in M": np.mean(is_cov),
    }


def fraction_correlation_matrix(matrices, tol=1e-12):
    is_diag_one = np.empty(matrices.shape[0], dtype=bool)
    is_sym = np.empty(matrices.shape[0], dtype=bool)
    is_pos_def = np.empty(matrices.shape[0], dtype=bool)
    is_corr = np.empty(matrices.shape[0], dtype=bool)

    for i, mat in enumerate(matrices):
        # Ones on the diagonal
        is_diag_one[i] = np.allclose(np.diag(mat), 1.0, atol=tol)

        # Symmetric
        is_sym[i] = np.allclose(mat, mat.T, atol=tol)

        # Positive definite
        eigvals = np.linalg.eigvalsh(mat)
        is_pos_def[i] = np.all(eigvals > tol)

        # correlation matrix
        is_corr[i] = is_diag_one[i] and is_sym[i] and is_pos_def[i]

    return {
        "Sym.": np.mean(is_sym),
        "Pos. def.": np.mean(is_pos_def),
        "Unit diag.": np.mean(is_diag_one),
        "in M": np.mean(is_corr),
    }


def project_on_SPD(matrices, eps=1e-8):
    """
    Projects a tensor of symmetric matrices onto the SPD cone with eigenvalues >= eps.

    Each matrix A is replaced by:
    A_proj = (1 - alpha) * A + alpha * I
    where alpha = (eps - lambda_min) / (1 - lambda_min) if lambda_min < eps else 0.

    Parameters:
        matrices: np.ndarray with shape (..., n, n)
        eps: minimum eigenvalue threshold (default: 1e-6)

    Returns:
        np.ndarray of same shape as matrices, with SPD-projected matrices
    """
    orig_shape = matrices.shape
    *batch_shape, n, n_ = orig_shape
    assert n == n_, "Matrices must be square."

    matrices_flat = matrices.reshape(-1, n, n)
    min_eigvals = np.linalg.eigvalsh(matrices_flat).min(axis=1)
    mask = min_eigvals < eps

    alphas = np.zeros_like(min_eigvals)
    alphas[mask] = (eps - min_eigvals[mask]) / (1 - min_eigvals[mask])

    # Identity matrix with broadcasting
    eye = np.eye(n)[None, :, :]
    matrices_spd = (1 - alphas)[:, None, None] * matrices_flat + alphas[
        :, None, None
    ] * eye

    return matrices_spd.reshape(*batch_shape, n, n)


def compute_quality_metrics(X_real, y_real, X_fake, y_fake):
    X_real_flat = X_real.reshape(len(X_real), -1)
    X_fake_flat = X_fake.reshape(len(X_fake), -1)

    if len(X_real_flat) != len(X_fake_flat):
        min_len = min(len(X_real_flat), len(X_fake_flat))
        X_real_flat = X_real_flat[:min_len]
        X_fake_flat = X_fake_flat[:min_len]
        y_real = y_real[:min_len]
        y_fake = y_fake[:min_len]

    prdc_metrics = compute_prdc(
        real_features=X_real_flat, fake_features=X_fake_flat, nearest_k=10
    )
    quality_metrics = {
        "Precision": prdc_metrics["precision"],
        "Recall": prdc_metrics["recall"],
        "Density": prdc_metrics["density"],
        "Coverage": prdc_metrics["coverage"],
        r"$\alpha$-precision": alpha_precision(
            X_real_flat, X_fake_flat, plot_curve=False
        ),
        r"$\beta$-recall": beta_recall(X_real_flat, X_fake_flat, plot_curve=False),
    }

    return quality_metrics


def compute_classification_metric(X_real, y_real, X_fake, y_fake, clf):
    if clf == "SVC":
        clf = SVC(
            kernel="rbf",
            C=1,
            probability=True,
            class_weight="balanced",
            gamma="scale",
            random_state=42,
            max_iter=5000,
        )
    elif clf == "LR":
        clf = LogisticRegressionCV(
            cv=5,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )
    elif clf == "dummy":
        clf = DummyClassifier()

    clf = make_pipeline(TangentSpace(metric="riemann"), clf)
    clf.fit(X_real, y_real)
    y_score_pred = clf.predict_proba(X_fake)
    y_pred = clf.predict(X_fake)

    return {
        "ROC-AUC": roc_auc_score(y_fake, y_score_pred[:, 1]),
        "Precision": precision_score(y_fake, y_pred),
        "Recall": recall_score(y_fake, y_pred),
        "F1": f1_score(y_fake, y_pred),
    }


def evaluate_metrics(dataset_method, group, split):
    rng = np.random.RandomState(0)

    path_results_dataset_method = PATH_RESULTS / dataset_method
    dataset = dataset_method.parts[0]
    group = dataset_method.parts[1]
    method = dataset_method.parts[2]

    print(f"Processing {dataset_method} - Split {split}...")

    def path_maker(end_path):
        return path_results_dataset_method / f"split_{split}_{end_path}.npy"

    cov_train = np.load(path_maker("covariances_train"))
    y_train = np.load(path_maker("conditionals_train"))
    cov_val = np.load(path_maker("covariances_val"))
    y_val = np.load(path_maker("conditionals_val"))
    generated_train = np.load(path_maker("covariances_generated_samples_train"))
    y_generated_train = np.load(path_maker("conditionals_generated_samples_train"))
    generated_val = np.load(path_maker("covariances_generated_samples_val"))
    y_generated_val = np.load(path_maker("conditionals_generated_samples_val"))
    training_time = float(np.load(path_maker("training_time")))
    sampling_time = float(np.load(path_maker("sampling_time")))

    frac_computation_constraints = (
        fraction_correlation_matrix if NORMALIZE else fraction_covariance_matrix
    )
    frac_constraints_train = frac_computation_constraints(cov_train, tol=1e-12)
    frac_constraints_val = frac_computation_constraints(cov_val, tol=1e-12)
    frac_constraints_gen = frac_computation_constraints(generated_train[-1], tol=1e-12)
    frac_constraints = [
        {
            "Dataset": dataset,
            "Method": method,
            "Group": group,
            "Split": split,
            "Subset": "Train",
            **frac_constraints_train,
        },
        {
            "Dataset": dataset,
            "Method": method,
            "Group": group,
            "Split": split,
            "Subset": "Val",
            **frac_constraints_val,
        },
        {
            "Dataset": dataset,
            "Method": method,
            "Group": group,
            "Split": split,
            "Subset": "Gen.",
            **frac_constraints_gen,
        },
    ]
    in_M = frac_constraints_gen["in M"] == 1.0

    if not in_M:
        generated_train_projected = project_on_SPD(generated_train)
        generated_val_projected = project_on_SPD(generated_val)

        # Verify projection succeeded
        frac_constraints_gen_train_projected = frac_computation_constraints(
            generated_train_projected[-1], tol=1e-12
        )
        frac_constraints_gen_val_projected = frac_computation_constraints(
            generated_val_projected[-1], tol=1e-12
        )
        assert frac_constraints_gen_train_projected["in M"] == 1.0, (
            "Projected generated training samples are not SPD."
        )
        assert frac_constraints_gen_val_projected["in M"] == 1.0, (
            "Projected generated validation samples are not SPD."
        )
    else:
        generated_train_projected = None
        generated_val_projected = None

    # Compute quality metrics
    quality_train_train = compute_quality_metrics(
        cov_train, y_train, cov_train, y_train
    )
    quality_train_val = compute_quality_metrics(cov_train, y_train, cov_val, y_val)
    quality_train_gen = compute_quality_metrics(
        cov_train, y_train, generated_train[-1], y_generated_train
    )
    quality_val_gen = compute_quality_metrics(
        cov_val, y_val, generated_val[-1], y_generated_val
    )
    quality_metrics = [
        {
            "Dataset": dataset,
            "Method": method,
            "Group": group,
            "Split": split,
            "Comparison": "Train vs Train",
            **quality_train_train,
        },
        {
            "Dataset": dataset,
            "Method": method,
            "Group": group,
            "Split": split,
            "Comparison": "Train vs Val",
            **quality_train_val,
        },
        {
            "Dataset": dataset,
            "Method": method,
            "Group": group,
            "Split": split,
            "Comparison": "Train vs Gen.",
            **quality_train_gen,
            "Train time (s)": training_time,
            "Sampling time (s)": sampling_time,
        },
        {
            "Dataset": dataset,
            "Method": method,
            "Group": group,
            "Split": split,
            "Comparison": "Val vs Gen.",
            **quality_val_gen,
        },
    ]
    if generated_train_projected is not None:
        quality_metrics += [
            {
                "Dataset": dataset,
                "Method": method + "_projected",
                "Group": group,
                "Split": split,
                "Comparison": "Train vs Gen.",
                **compute_quality_metrics(
                    cov_train, y_train, generated_train_projected[-1], y_generated_train
                ),
                "Train time (s)": training_time,
                "Sampling time (s)": sampling_time,
            },
            {
                "Dataset": dataset,
                "Method": method + "_projected",
                "Group": group,
                "Split": split,
                "Comparison": "Val vs Gen.",
                **compute_quality_metrics(
                    cov_val, y_val, generated_val_projected[-1], y_generated_val
                ),
            },
        ]

    # Compute classification metrics
    baseline = compute_classification_metric(cov_train, y_train, cov_val, y_val, "LR")
    gan_train = [
        {
            "Dataset": dataset,
            "Method": method,
            "Group": group,
            "Split": split,
            "Comparison": "Train vs Val",
            **baseline,
            "Train time (s)": training_time,
            "Sampling time (s)": sampling_time,
        }
    ]
    gan_test = []  # Initialize gan_test list

    if frac_constraints_gen["Pos. def."] == 1.0:
        gan_train_scores = compute_classification_metric(
            generated_train[-1], y_generated_train, cov_val, y_val, "LR"
        )
        gan_test_scores = compute_classification_metric(
            cov_train, y_train, generated_train[-1], y_generated_train, "LR"
        )
        gan_train += [
            {
                "Dataset": dataset,
                "Method": method,
                "Group": group,
                "Split": split,
                "Comparison": "Gen vs Val",
                **gan_train_scores,
                "Train time (s)": training_time,
                "Sampling time (s)": sampling_time,
            }
        ]
        gan_test += [
            {
                "Dataset": dataset,
                "Method": method,
                "Group": group,
                "Split": split,
                "Comparison": "Train vs Gen",
                **gan_test_scores,
                "Train time (s)": training_time,
                "Sampling time (s)": sampling_time,
            }
        ]
    else:
        # Original generated samples are not SPD, so record NaNs for the original method
        gan_scores_nan = {
            "ROC-AUC": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
        }
        gan_train += [
            {
                "Dataset": dataset,
                "Method": method,
                "Group": group,
                "Split": split,
                "Comparison": "Gen vs Val",
                **gan_scores_nan,
                "Train time (s)": training_time,
                "Sampling time (s)": sampling_time,
            }
        ]
        gan_test += [
            {
                "Dataset": dataset,
                "Method": method,
                "Group": group,
                "Split": split,
                "Comparison": "Train vs Gen",
                **gan_scores_nan,
                "Train time (s)": training_time,
                "Sampling time (s)": sampling_time,
            }
        ]

        # Compute and record scores for the projected generated samples
        gan_train_scores = compute_classification_metric(
            generated_train_projected[-1], y_generated_train, cov_val, y_val, "LR"
        )
        gan_test_scores = compute_classification_metric(
            cov_train, y_train, generated_train_projected[-1], y_generated_train, "LR"
        )
        gan_train += [
            {
                "Dataset": dataset,
                "Method": method + "_projected",
                "Group": group,
                "Split": split,
                "Comparison": "Gen vs Val",
                **gan_train_scores,
                "Train time (s)": training_time,
                "Sampling time (s)": sampling_time,
            }
        ]
        gan_test += [
            {
                "Dataset": dataset,
                "Method": method + "_projected",
                "Group": group,
                "Split": split,
                "Comparison": "Train vs Gen",
                **gan_test_scores,
                "Train time (s)": training_time,
                "Sampling time (s)": sampling_time,
            }
        ]

    print(f"Metrics for {dataset_method} - Split {split} computed.")

    return frac_constraints, quality_metrics, gan_train, gan_test


if __name__ == "__main__":
    path_figures_modality = PATH_FIGURES / EXPE
    path_figures_modality.mkdir(parents=True, exist_ok=True)

    results_dict = gather_results_structure(PATH_RESULTS, DATASETS)

    tasks = [
        (dataset, group, method, n_splits)
        for dataset, groups in results_dict.items()
        for group, methods in groups.items()
        for method, n_splits in methods.items()
    ]

    all_metrics = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_metrics)(Path(dataset) / group / method, group, split)
        for (dataset, group, method, n_splits) in tasks
        for split in range(n_splits)
    )

    # 1) flatten the results
    frac_constraints = [d for frac, *_ in all_metrics for d in frac]
    quality_metrics = [d for _, sublist, *_ in all_metrics for d in sublist]
    gan_train_metrics = [d for _, _, gt, _ in all_metrics for d in gt]
    gan_test_metrics = [d for _, _, _, gs in all_metrics for d in gs]

    # 2) write CSVs
    pd.DataFrame(frac_constraints).to_csv(
        path_figures_modality / "fraction_constraints.csv",
        index=False,
        float_format="%.3f",
    )
    pd.DataFrame(quality_metrics).to_csv(
        path_figures_modality / "quality_metrics.csv", index=False, float_format="%.3f"
    )
    pd.DataFrame(gan_train_metrics).to_csv(
        path_figures_modality / "gan_train_metrics.csv",
        index=False,
        float_format="%.3f",
    )
    pd.DataFrame(gan_test_metrics).to_csv(
        path_figures_modality / "gan_test_metrics.csv", index=False, float_format="%.3f"
    )
