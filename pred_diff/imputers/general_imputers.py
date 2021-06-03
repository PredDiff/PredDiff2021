import numpy as np

from .imputer_base import ImputerBase

from typing import Optional

class TrainSetImputer(ImputerBase):
    """
    imputer just inserts randomly sampled training samples
    """

    def __init__(self, train_data: np.ndarray, **kwargs):
        super().__init__(train_data=train_data)
        kwargs["label_encode"] = False
        kwargs["standard_scale"] = False
        self.imputer_name = 'TrainSetImputer'

    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        # returns only imputation for a single mask
        np_test = np.array(test_data)
        n_samples = np_test.shape[0]
        rng = np.random.default_rng()
        rng.choice(self.train_data, n_samples, replace=True)  # replace: multiple occurrences are allowed
        res = rng.choice(self.train_data, n_samples * n_imputations, replace=True).copy()
        imputations = res.reshape(n_imputations, n_samples, *mask_impute.shape)
        return imputations, None


class GaussianNoiseImputer(ImputerBase):
    """
    Adds gaussian white noise onto the test samples.
    """
    def __init__(self, train_data: np.ndarray, sigma: Optional[np.ndarray] = None, **kwargs):
        """sigma: array containing the deviation for every feature dimension"""
        super().__init__(train_data=train_data)
        self.imputer_name = 'GaussianNoiseImputer'

        self.sigma = self.train_data[:300].std(axis=0) if sigma is None else sigma        # variance for gaussian noise
        assert np.alltrue(train_data[0].shape == self.sigma.shape), 'incorrect shape for variance sigma'

    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations=100):
        imputations = np.array([test_data.copy() for _ in range(n_imputations)])
        noise = np.random.randn(imputations.size).reshape(imputations.shape)
        return imputations + self.sigma * noise, None
