from pathlib import Path
import pickle
import numpy as np
import torch
import tabulate
from sklearn.preprocessing import LabelEncoder
import urllib.request
import subprocess

from moabb.datasets import BNCI2014_002, BNCI2015_001
from moabb.paradigms import MotorImagery


PATH_CACHE = Path("data")

FMRI_DATASETS = ["abide", "adni", "oasis3"]
EEG_DATASETS = ["bnci2014_002", "bnci2015_001"]

FMRI_URL = "https://osf.io/h7sw5/download"


def load_data(dataset, atlas, rng, verbose=True):
    dataset = dataset.lower()

    if dataset in FMRI_DATASETS:
        path_atlas = Path(f"{PATH_CACHE}_atlas_{atlas}")
        path_dataset = path_atlas / f"{dataset}_X_y.pkl"

        # Automatic Download and Extraction if the file doesn't exist
        if not path_dataset.exists():
            print(f"Dataset '{dataset}' with atlas '{atlas}' not found locally.")
            print("Attempting to download from OSF...")

            # The parent directory of the final file might not exist yet
            path_dataset.parent.mkdir(parents=True, exist_ok=True)

            zip_path = Path(__file__).parent / f"data_atlas_{atlas}.zip"

            try:
                if not zip_path.exists():
                    print(f"Downloading data from {FMRI_URL}...")
                    urllib.request.urlretrieve(FMRI_URL, zip_path)
                    print("Download complete.")
                else:
                    print(f"Found existing zip at '{zip_path}', skipping download.")

                print("Extracting files with system unzip...")
                subprocess.run(
                    ["unzip", "-o", str(zip_path), "-d", str(Path(__file__).parent)],
                )
                print(f"Extraction complete. Data should now be in '{path_dataset.parent}'.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download or extract dataset '{dataset}' "
                    f"with atlas '{atlas}' from {FMRI_URL}: {e}"
                )

        # Now, load the data (either it was already there or just downloaded)
        path = (Path(__file__).parent / path_dataset).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found after download attempt. Expected at: {path}"
            )

        with open(path, "rb") as f:
            df = pickle.load(f)

        # Print infos about the dataset
        if verbose:
            print(
                tabulate.tabulate(
                    [
                        ["Dataset", dataset],
                        ["Atlas", atlas],
                        ["Number of subjects", df["SubjectID"].nunique()],
                        ["Number of time series", len(df)],
                        [
                            "Number of time series per subject",
                            round(len(df) / df["SubjectID"].nunique(), 1),
                        ],
                        ["Number of brain regions", df["TimeSeries"].iloc[0].shape[1]],
                        [
                            "Number of classes (diagnosis)",
                            df["Diagnosis"].nunique() if "Diagnosis" in df else "N/A",
                        ],
                        [
                            "Age range",
                            f"min: {df['Age'].min()}, max: {df['Age'].max()}",
                        ],
                        ["Loaded from", path],
                    ]
                )
            )

        df.reset_index(drop=True, inplace=True)

        # Shuffle dataframe
        idx = rng.permutation(np.arange(len(df)))
        df = df.iloc[idx].reset_index(drop=True)
        df = df.reset_index(drop=True)

        # Diagnosis: control (0) vs other (>0)
        df["Diagnosis"] = (df["Diagnosis"] != 0).astype(int)

        if "Group" in df.columns:
            df = df.drop(columns=["Group"])

        ts = df["TimeSeries"].values
        y = df["Diagnosis"].values
        groups = df["SubjectID"].values

    else:
        assert dataset in EEG_DATASETS, f"Dataset {dataset} not supported"

        if dataset == "bnci2014_002":
            ds = BNCI2014_002()
        elif dataset == "bnci2015_001":
            ds = BNCI2015_001()
        else:
            raise ValueError(f"EEG dataset {dataset} not yet implemented")

        paradigm = MotorImagery()
        data = paradigm.get_data(ds)

        if dataset == "bnci2014_002":
            groups = data[2]["run"].values
            mask = np.array([True] * len(groups))
        elif dataset == "bnci2015_001":
            groups = data[2]["session"].values
            # Remove 3rd session
            mask = groups != "2C"
        else:
            raise ValueError(f"EEG dataset {dataset} not yet implemented")

        ts = data[0].swapaxes(-1, -2)[mask]
        y = data[1][mask]
        y = LabelEncoder().fit_transform(y)
        groups = LabelEncoder().fit_transform(groups)
        groups = groups[mask]

        if dataset == "bnci2014_002":
            mask = groups != 0
            ts, y, groups = ts[mask], y[mask], groups[mask]
            groups = groups - 1

        # Print infos
        if verbose:
            print(
                tabulate.tabulate(
                    [
                        ["Dataset", dataset],
                        ["Number of sessions", len(np.unique(groups))],
                        ["Number of time series", len(ts)],
                        ["Time series shape", ts[0].shape],
                        ["Number of classes", len(np.unique(y))],
                        ["Loaded using", "MOABB"],
                    ]
                )
            )

        # Shuffle
        idx = rng.permutation(np.arange(len(ts)))
        ts, y, groups = ts[idx], y[idx], groups[idx]

    return ts, y, groups


class FastDataloader:
    def __init__(
        self,
        x1,
        y1,
        batch_size,
        time_sampler=None,
        prior=None,
        shuffle=False,
        drop_last=False,
    ):
        """
        :param x1: array-like of shape (n_samples, ...)
        :param y1: array-like of shape (n_samples,)
        :param batch_size: batch size
        :param time_sampler: callable that takes batch_size and returns (batch_size,)
        :param prior: callable that takes y_cond and returns (batch_size, ...)
        """
        self.x1 = x1
        self.y1 = y1
        self.time_sampler = time_sampler
        self.prior = prior
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset_len = len(x1)

    def __iter__(self):
        if self.shuffle:
            self.idxs = torch.randperm(self.dataset_len)
        else:
            self.idxs = torch.arange(self.dataset_len)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        end = self.i + self.batch_size
        if self.drop_last and end > self.dataset_len:
            raise StopIteration

        idx = self.idxs[self.i : end]
        x1_batch = self.x1[idx]
        y1_batch = self.y1[idx]

        bs = len(idx)
        if self.time_sampler is not None:
            t_batch = self.time_sampler(bs)
        if self.prior is not None:
            x0_batch = self.prior(y1_batch)

        self.i = end

        to_return = list()
        if self.time_sampler is not None:
            to_return.append(t_batch)
        if self.prior is not None:
            to_return.append(x0_batch)
        to_return.append(x1_batch)
        to_return.append(y1_batch)

        return tuple(to_return)

    def __len__(self):
        n = self.dataset_len
        return (
            n // self.batch_size
            if self.drop_last
            else (n + self.batch_size - 1) // self.batch_size
        )
