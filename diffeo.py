import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


class DiffeoTransform(BaseEstimator, TransformerMixin):
    """
    Applies a diffeomorphism to SPD or correlation matrices
    and returns a vectorized form for use in machine learning pipelines.

    Supported diffeomorphisms:
    - 'lower_triangular': SPD → lower triangular matrix
    - 'logeuclidean': SPD → symmetric (via matrix logarithm)
    - 'logcholesky': SPD → lower triangular (via log of Cholesky)
    - 'strict_lower_triangular': correlation → strictly lower triangular matrix
    - 'corrcholesky': correlation → lower triangular with unit diag
    - 'pyriemann_pca': correlation → tangent space + PCA

    Vectorization:
    - For SPD: lower triangle including diagonal.
    - For correlation: strictly lower triangle (diagonal excluded).
    """

    def __init__(self, diffeo="logeuclidean"):
        self.diffeo = diffeo.lower()
        self._cov_diffeos = ["logeuclidean", "logcholesky", "lower_triangular"]
        self._corr_diffeos = [
            "corrcholesky",
            "pyriemann_pca",
            "strict_lower_triangular",
        ]
        self._all_diffeos = self._cov_diffeos + self._corr_diffeos
        if self.diffeo not in self._all_diffeos:
            raise ValueError(f"Unsupported diffeomorphism: {self.diffeo}")

    def fit(self, X, y=None):
        X = self._ensure_batched_matrix(X)
        self._check_input_domain(X)
        if self.diffeo == "pyriemann_pca":
            self.pipeline_ = make_pipeline(
                TangentSpace(metric="riemann", tsupdate=False), PCA(n_components=300)
            )
            self.pipeline_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._ensure_batched_matrix(X)
        self._check_input_domain(X)

        if self.diffeo == "logeuclidean":
            X_tilde = self._logm(X)
            return self._vectorize(X_tilde)
        elif self.diffeo == "logcholesky":
            X_tilde = self._log_cholesky(X)
            return self._vectorize(X_tilde)
        elif self.diffeo == "corrcholesky":
            X_tilde = self._corr_cholesky(X)
            return self._vectorize(X_tilde)
        elif self.diffeo == "pyriemann_pca":
            return self.pipeline_.transform(X)
        elif self.diffeo in ["lower_triangular", "strict_lower_triangular"]:
            return self._vectorize(X)

    def inverse_transform(self, X_vec: np.ndarray) -> np.ndarray:
        X_vec = self._ensure_batched_vector(X_vec)

        if self.diffeo == "logeuclidean":
            X_mat = self._devectorize(X_vec)
            diag = np.zeros_like(X_mat)
            idx = np.arange(X_mat.shape[-1])
            diag[..., idx, idx] = X_mat[..., idx, idx]
            X_mat_lower = X_mat - diag
            X_mat = X_mat + np.swapaxes(X_mat_lower, -1, -2)
            return self._expm(X_mat)
        elif self.diffeo == "logcholesky":
            X_mat = self._devectorize(X_vec)
            return self._log_cholesky_inv(X_mat)
        elif self.diffeo == "corrcholesky":
            X_mat = self._devectorize(X_vec)
            X_mat = X_mat + np.eye(X_mat.shape[-1])[..., None, :, :]
            return self._corr_cholesky_inv(X_mat)
        elif self.diffeo == "pyriemann_pca":
            X_recov = self.pipeline_.inverse_transform(X_vec)
            diag = np.sqrt(np.diagonal(X_recov, axis1=-2, axis2=-1))
            return X_recov / (diag[..., None] * diag[..., None, :])
        elif self.diffeo == "lower_triangular":
            X_mat = self._devectorize(X_vec)
            X_mat_strict = np.tril(X_mat, k=-1)
            X_mat = X_mat + X_mat_strict.swapaxes(-1, -2)
            return X_mat
        elif self.diffeo == "strict_lower_triangular":
            X_mat = self._devectorize(X_vec)
            X_mat = X_mat + X_mat.swapaxes(-1, -2)
            X_mat = X_mat + np.eye(X_mat.shape[-1])[..., None, :, :]
            return X_mat

    # ---- Internal Methods ----

    def _ensure_batched_matrix(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = X[None, ...]
        if X.ndim < 3:
            raise ValueError("Expected at least 3D array.")
        return X

    def _ensure_batched_vector(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[None, ...]
        if X.ndim < 2:
            raise ValueError("Expected at least 2D array.")
        return X

    def _check_input_domain(self, X: np.ndarray):
        diag = np.diagonal(X, axis1=-2, axis2=-1)
        is_corr = np.allclose(diag, 1.0, atol=1e-4)
        if self.diffeo in self._corr_diffeos and not is_corr:
            warnings.warn(
                f"{self.diffeo} expects correlation matrices (unit diag)", UserWarning
            )
        if self.diffeo in self._cov_diffeos and is_corr:
            warnings.warn(
                f"{self.diffeo} expects SPD matrices (non-unit diag)", UserWarning
            )

    def _infer_matrix_dim(self, v: np.ndarray) -> int:
        n = v.shape[-1]
        if self.diffeo in self._corr_diffeos:
            d = int((1 + (1 + 8 * n) ** 0.5) / 2)
        else:
            d = int((-1 + (1 + 8 * n) ** 0.5) / 2)
        return d

    def _logm(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                U @ np.diag(np.log(w)) @ U.T
                for w, U in map(np.linalg.eigh, X.reshape(-1, *X.shape[-2:]))
            ]
        ).reshape(*X.shape)

    def _expm(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                U @ np.diag(np.exp(w)) @ U.T
                for w, U in map(np.linalg.eigh, X.reshape(-1, *X.shape[-2:]))
            ]
        ).reshape(*X.shape)

    def _log_cholesky(self, X: np.ndarray) -> np.ndarray:
        L = np.linalg.cholesky(X)
        log_diag = np.log(np.diagonal(L, axis1=-2, axis2=-1))
        L_out = np.tril(L, k=-1)
        return L_out + np.einsum("...i,ij->...ij", log_diag, np.eye(L.shape[-1]))

    def _log_cholesky_inv(self, Llog: np.ndarray) -> np.ndarray:
        diag = np.exp(np.diagonal(Llog, axis1=-2, axis2=-1))
        L = np.tril(Llog, -1) + np.einsum(
            "...i,ij->...ij", diag, np.eye(Llog.shape[-1])
        )
        return L @ np.swapaxes(L, -1, -2)

    def _corr_cholesky(self, X: np.ndarray) -> np.ndarray:
        L = np.linalg.cholesky(X)
        diag_L = np.diagonal(L, axis1=-2, axis2=-1)
        inv_diag = 1.0 / diag_L
        return inv_diag[..., None] * L

    def _corr_cholesky_inv(self, L: np.ndarray) -> np.ndarray:
        LLt = L @ np.swapaxes(L, -1, -2)
        diag = np.diagonal(LLt, axis1=-2, axis2=-1)
        inv_sqrt_diag = 1.0 / np.sqrt(diag)
        Dinv = inv_sqrt_diag[..., None] * inv_sqrt_diag[..., None, :]
        return Dinv * LLt

    def _vectorize(self, X: np.ndarray) -> np.ndarray:
        """Vectorizes the matrix X.
        For SPD: lower triangle including diagonal to vector.
        For correlation: strictly lower triangle (diagonal excluded) to vector.
        """
        d = X.shape[-1]
        if self.diffeo in self._corr_diffeos:
            idx = np.tril_indices(d, k=-1)
        else:
            idx = np.tril_indices(d, k=0)
        return X[..., idx[0], idx[1]]

    def _devectorize(self, v: np.ndarray) -> np.ndarray:
        """Devectorizes the vector v.
        For SPD: vector to lower triangle including diagonal.
        For correlation: vector to strictly lower triangle (diagonal excluded).
        """
        d = self._infer_matrix_dim(v)
        shape = v.shape[:-1]
        M = np.zeros((*shape, d, d))
        if self.diffeo in self._corr_diffeos:
            idx = np.tril_indices(d, k=-1)
        else:
            idx = np.tril_indices(d, k=0)
        M[..., idx[0], idx[1]] = v
        return M


class DiffeomorphicMixin:
    """Mixin providing diffeomorphic preprocessing for matrix-valued data."""

    def __init__(self, diffeomorphism: str | None = None):
        self._diffeo_name: str | None = None
        self._diffeo_transform: DiffeoTransform | None = None
        self._feature_scaler: StandardScaler | None = None
        self._matrix_dim: int | None = None
        self._vector_dim: int | None = None
        if diffeomorphism is not None:
            self.set_diffeomorphism(diffeomorphism)

    def set_diffeomorphism(self, diffeomorphism: str | None) -> None:
        """Configure the diffeomorphism used to embed SPD/correlation matrices."""
        self._diffeo_name = diffeomorphism
        if diffeomorphism is None:
            self._diffeo_transform = None
            self._feature_scaler = None
            self._matrix_dim = None
            self._vector_dim = None
        else:
            self._diffeo_transform = DiffeoTransform(diffeo=diffeomorphism)
            self._feature_scaler = StandardScaler()

    def _ensure_vector(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim > 2:
            return X.reshape(X.shape[0], -1)
        return X

    def _fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        if self._diffeo_transform is None:
            return self._ensure_vector(X)
        X = np.asarray(X)
        self._matrix_dim = X.shape[-1]
        X_vec = self._diffeo_transform.fit_transform(X)
        self._vector_dim = X_vec.shape[-1]
        X_scaled = self._feature_scaler.fit_transform(X_vec)
        return X_scaled

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        if self._diffeo_transform is None:
            return self._ensure_vector(X)
        X_vec = self._diffeo_transform.transform(X)
        return self._feature_scaler.transform(X_vec)

    def _inverse_transform_features(self, Z: np.ndarray) -> np.ndarray:
        if self._diffeo_transform is None:
            return np.asarray(Z)
        Z = np.asarray(Z)
        original_shape = Z.shape[:-1]
        Z_vec = Z.reshape(-1, Z.shape[-1])
        Z_vec = self._feature_scaler.inverse_transform(Z_vec)
        mats = self._diffeo_transform.inverse_transform(Z_vec)
        mat_dim = mats.shape[-1]
        return mats.reshape(*original_shape, mat_dim, mat_dim)

    @property
    def diffeomorphism_(self) -> str | None:
        return self._diffeo_name

    def is_diffeo(self):
        """Check if the transform is a diffeomorphism.

        Returns:
            bool: True if the transform is a diffeomorphism, False otherwise.
        """
        if self.diffeo in ["logeuclidean", "logcholesky", "corrcholesky"]:
            return True
        elif self.diffeo in [
            "lower_triangular",
            "strict_lower_triangular",
            "pyriemann_pca",
        ]:
            return False
        else:
            raise ValueError(f"Unknown diffeomorphism: {self.diffeo}")


if __name__ == "__main__":
    from itertools import product

    def sample_cov(n, d):
        A = np.random.randn(n, d, d)
        return A @ np.transpose(A, (0, 2, 1)) + 1e-1 * np.eye(d)

    def sample_corr(n, d):
        C = sample_cov(n, d)
        std = np.sqrt(np.diagonal(C, axis1=-2, axis2=-1))
        return C / (std[..., None] * std[..., None, :])

    n_list = [1, 2, 3]
    d_list = [2, 3, 4]

    for n, d in product(n_list, d_list):
        print(f"Testing with n={n}, d={d}")

        for name, sampler in [
            ("logeuclidean", sample_cov),
            ("logcholesky", sample_cov),
            ("corrcholesky", sample_corr),
        ]:
            X = sampler(n, d)
            tf = DiffeoTransform(diffeo=name)
            Z = tf.transform(X)
            X_inv = tf.inverse_transform(Z)

            assert np.allclose(X, X_inv, atol=1e-5, rtol=1e-3), (
                f"Inverse transform failed for {name} with n={n}, d={d}"
            )
            print(f"[{name}] vectorized shape: {Z.shape}")
            print(
                f"[{name}] max inverse error: {np.linalg.norm(X - X_inv, axis=(-2, -1)).max():.2e}"
            )

            # # plot side by side X and X_inv
            # import matplotlib.pyplot as plt
            # from nilearn.plotting import plot_matrix

            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # vmin, vmax = min(np.min(X), np.min(X_inv)), max(np.max(X), np.max(X_inv))
            # plot_matrix(X[0], axes=axs[0], colorbar=True, vmin=vmin, vmax=vmax)
            # axs[0].set_title('Original Matrix')
            # plot_matrix(X_inv[0], axes=axs[1], colorbar=True, vmin=vmin, vmax=vmax)
            # axs[1].set_title('Inverse Transformed Matrix')
            # plt.show()
