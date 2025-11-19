import numpy as np

from sklearn.covariance import OAS
from joblib import delayed, Parallel


def cov_est(ts, normalize=True, n_jobs=1):
    """Estimate the covariance matrices of a list of time series.

    Args:
        ts: list of np.arrays, each representing a time series.
        normalize: bool, whether to normalize the covariance matrices to
            have unit diagonal (i.e., correlation matrices).

    Returns:
        np.array: 3D array containing correlation matrices.
    """
    if (type(ts) not in (list, tuple)) and (type(ts) is np.ndarray and ts.ndim == 2):
        ts = [ts]

    def _cov_est(ts):
        return OAS(store_precision=False).fit(ts).covariance_

    if n_jobs == 1:
        cov = np.array([_cov_est(t) for t in ts])
    else:
        cov = Parallel(n_jobs=n_jobs)(delayed(_cov_est)(t) for t in ts)
        cov = np.array(cov)

    if normalize:
        std_devs = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        normalization = std_devs[:, :, None] * std_devs[:, None, :]
        cov = cov / normalization

    return cov
