import numpy as np
from sklearn.covariance import OAS

from diffeo import DiffeomorphicMixin


class ClassConditionalGaussianPrior:
    def __init__(self, random_state=None):
        if isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)
        self.means_ = None
        self.covariances_cholesky_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimate one Gaussian per class.
        Args:
            X: (n_samples, dim) features
            y: (n_samples,) integer labels
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.dim_ = X.shape[1]

        self.means_ = []
        self.covariances_cholesky_ = []

        for cls in self.classes_:
            X_cls = X[y == cls]

            mean = X_cls.mean(axis=0)
            self.means_.append(mean)

            cov = OAS(store_precision=False).fit(X_cls).covariance_
            L = np.linalg.cholesky(cov)
            self.covariances_cholesky_.append(L)

        self.means_ = np.stack(self.means_)
        self.covariances_cholesky_ = np.stack(self.covariances_cholesky_)

    def sample(self, y_cond: np.ndarray) -> np.ndarray:
        """
        Sample from the corresponding Gaussian for each class.
        Args:
            y_cond: (n_samples,) integer labels
        Returns:
            samples: (n_samples, dim) samples from the corresponding Gaussian
        """
        if self.means_ is None or self.covariances_cholesky_ is None:
            raise RuntimeError("Call `fit` before sampling.")

        samples = np.empty((len(y_cond), self.means_.shape[-1]))

        for i, k in enumerate(np.unique(y_cond)):
            mean = self.means_[k]
            cov_cholesky = self.covariances_cholesky_[k]
            idx = y_cond == k
            samples[idx] = (
                self.rng.randn(np.sum(idx), self.dim_) @ cov_cholesky.T + mean
            )

        # Return np.newaxis to ensure the output shape is (n_steps, ...)
        return samples[np.newaxis, ...]


class DiffeoGauss(DiffeomorphicMixin):
    """Baseline Gaussian model operating in a diffeomorphic latent space."""

    def __init__(self, config: dict):
        self.config = dict(config)
        rng = self.config.get("RNG")
        diffeo = self.config.get("DIFFEO")
        super().__init__(diffeomorphism=diffeo)
        self._prior = ClassConditionalGaussianPrior(random_state=rng)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        X_proj = self._fit_transform_features(X)
        self._prior.fit(X_proj, y)
        return None

    def sample(self, y_cond: np.ndarray) -> np.ndarray:
        samples = self._prior.sample(y_cond).squeeze(0)
        samples = samples[np.newaxis, ...]
        return self._inverse_transform_features(samples)

    def set_diffeomorphism(self, diffeomorphism: str | None) -> None:
        self.config["DIFFEO"] = diffeomorphism
        super().set_diffeomorphism(diffeomorphism)
        self._prior = ClassConditionalGaussianPrior(random_state=self.config.get("RNG"))


if __name__ == "__main__":
    # Example usage
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)
    y_cond = np.random.randint(0, 2, size=20)

    prior = ClassConditionalGaussianPrior()
    prior.fit(X, y)
    samples = prior.sample(y_cond)
    assert samples.shape == (20, 10)
