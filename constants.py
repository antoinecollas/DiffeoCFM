import argparse
from pathlib import Path
import torch
import numpy as np

from fm import DiffeoCFM
from spd_fm import SPDConditionalFlowMatching
from gaussian import DiffeoGauss


# fix seeds
torch.manual_seed(42)
np.random.seed(42)

# debug mode
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument(
    "--modality",
    type=str,
    default="fmri",
    choices=["fmri", "eeg"],
    help="Modality to use: 'fmri' or 'eeg'.",
)
args = parser.parse_args()
DEBUG = args.debug
EXPE = args.modality

# device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using gpu")
else:
    DEVICE = "cpu"
    print("Using cpu.")


TEST_SIZE = 0.1
if EXPE == "fmri":
    DATASETS = ["abide", "adni", "oasis3"]
    ATLAS = "msdl"
    NORMALIZE = True
    EPOCHS = 200
    FACTOR_LR = 1
    HIDDEN_DIM = [512]
    LR = 0.001
else:
    assert EXPE == "eeg", "EXPE must be 'fmri' or 'eeg'."
    DATASETS = ["bnci2014_002", "bnci2015_001"]
    ATLAS = None
    NORMALIZE = False
    EPOCHS = 2000
    FACTOR_LR = 0.1
    HIDDEN_DIM = [512]
    LR = 0.001

if DEBUG:
    N_JOBS = 1
    EPOCHS = 10
    N_SPLITS = 2
    WARMUP_EPOCHS = 5
else:
    N_JOBS = -1
    N_SPLITS = 10 if EXPE == "fmri" else 5
    WARMUP_EPOCHS = 10

CONFIG_FM = {
    "FM_TYPE": "classic",
    "WARMUP_EPOCHS": WARMUP_EPOCHS,
    "FACTOR_LR": FACTOR_LR,
    "LR": LR,
    "BATCH_SIZE": 64,
    "EPOCHS": EPOCHS,
    "HIDDEN_DIM": HIDDEN_DIM,
    "PRINT_EVERY": 100,
    "T_GRID": torch.linspace(0, 1, 6, device=DEVICE, dtype=torch.float64),
    "DEVICE": DEVICE,
    "RNG": np.random.RandomState(42),
}

CONFIG_SPD_CFM = CONFIG_FM.copy()
CONFIG_SPD_CFM["LR"] = 1e-4
CONFIG_SPD_CFM["HIDDEN_DIM"] = 6 * [512]
CONFIG_SPD_CFM["WARMUP_EPOCHS"] = 200

# helper to clone configs with fresh RNG and diffeomorphism
def _make_config(base_cfg: dict, diffeo: str | None = None) -> dict:
    cfg = base_cfg.copy()
    cfg["RNG"] = np.random.RandomState(42)
    if diffeo is None and "DIFFEO" in cfg:
        cfg.pop("DIFFEO")
    if diffeo is not None:
        cfg["DIFFEO"] = diffeo
    return cfg


# methods
METHODS = list()

if not NORMALIZE:
    METHODS.append(
        {
            "diffeo": None,
            "model": SPDConditionalFlowMatching(_make_config(CONFIG_SPD_CFM, None)),
        }
    )

METHODS = METHODS + [
    {
        "diffeo": "corrcholesky" if NORMALIZE else "logeuclidean",
        "model": DiffeoGauss(
            {
                "RNG": np.random.RandomState(42),
                "DIFFEO": "corrcholesky" if NORMALIZE else "logeuclidean",
            }
        ),
    },
    {
        "diffeo": "strict_lower_triangular" if NORMALIZE else "lower_triangular",
        "model": DiffeoCFM(
            _make_config(
                CONFIG_FM,
                "strict_lower_triangular" if NORMALIZE else "lower_triangular",
            )
        ),
    },
    {
        "diffeo": "corrcholesky" if NORMALIZE else "logeuclidean",
        "model": DiffeoCFM(
            _make_config(
                CONFIG_FM,
                "corrcholesky" if NORMALIZE else "logeuclidean",
            )
        ),
    },
]


# paths
PATH_RESULTS = Path("results")
PATH_RESULTS.mkdir(parents=True, exist_ok=True)
PATH_FIGURES = Path("figures")
PATH_FIGURES.mkdir(parents=True, exist_ok=True)
