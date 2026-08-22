import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from typing import Union

logger = logging.getLogger(__name__)


def canonicalize_geom(interatom: np.ndarray, molecule: str) -> np.ndarray:
    """
    Select only symmetry unique points from the dataset.
    Depends on molucule and the data must be in specific form.
    - h2o: (H - O - H)

        (rHH, rOH1, rOH2) -> (rHH, rOH_long, rOH_short)

    - n2o: (N1 - N2 - O)

        (rN1N2, rN2O, rN1O) -> (rN1N2, rN2O, rN1O)
        (Identity)

    Args:
        interatom (np.ndarray): Data to be transformed.
        molecule (str): Molecule the data belongs to. Options: `'h2o'`,`'n2o'`.
    """
    if molecule == "h2o":
        rHH = interatom[:, 0]
        r1 = interatom[:, 1]
        r2 = interatom[:, 2]

        r_short = np.minimum(r1, r2)
        r_long = np.maximum(r1, r2)

        return np.column_stack((rHH, r_long, r_short))

    elif molecule == "n2o":
        return interatom
    else:
        raise ValueError("Invalid molecule: %s", molecule)


def morse(X: np.ndarray, morse_param: int = 1.0) -> np.ndarray:
    """Do morse transform (y=exp(-a*x))."""
    return np.exp(-morse_param * X)


def pips(X: np.ndarray, molecule: str) -> np.ndarray:
    """
    Do PIP transform of data `X`. Depends on `molecule`. Data must be in correct
    order.
    - h2o:

        (rHH, rOH, rOH) -> (rHH, rOH1+rOH2, rOH1**2+rOH2**2)

    - n2o:

        (rN1N2, rN2O, rN1O) -> (like water)
    Args:
        X (np.ndarray): Data to be transformed.
        molecule (str): Molecule the data belongs to. Options: `'h2o'`,`'n2o'`.
    """
    if molecule in ["h2o", "n2o"]:
        # water (rHH, rOH, rOH)
        X_new = np.zeros_like(X)
        X_new[:, 0] = X[:, 0]
        X_new[:, 1] = X[:, 1] + X[:, 2]
        X_new[:, 2] = X[:, 1] ** 2 + X[:, 2] ** 2
    else:
        raise ValueError(f"Invalid choice of molecule: {molecule}")
    return X_new


class Preprocessor:
    """
    Preprocesses data and targets including canonicalization, morse, pips,
    machine learning scaling.

    Preprocessor must do transformations in this order:
    1. Remove redundant geometries transformation
    2. Morse transform
    3. Pip Transform
    4. Standard/MinMax scaling

    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.scaler_X = None
        self.scaler_y = None
        self.fitted_X = False
        self.fitted_y = False

        logger.info("Preprocessor initialized.")

    def _create_scaler(
        self, scaler_type: str = "std"
    ) -> Union[StandardScaler, MinMaxScaler]:
        """
        Create a feature scaler.

        Args:
            scaler_type : {'std', 'minmax01', 'minmax11'}, default='std'
                Type of scaling to apply:

                - ``'std'`` — standardization using mean and variance
                (``StandardScaler``)
                - ``'minmax01'`` — rescale features to the interval [0, 1]
                - ``'minmax11'`` — rescale features to the interval [-1, 1]

        Returns:
            StandardScaler or MinMaxScaler: Configured scaler.
        """

        scalers = {
            "std": StandardScaler,
            "minmax01": lambda: MinMaxScaler(feature_range=(0, 1)),
            "minmax11": lambda: MinMaxScaler((-1, 1)),
        }

        if scaler_type not in scalers:
            raise ValueError(f"Invalid scaler_type '{scaler_type}'")

        return scalers[scaler_type]()

    def _physical_transform(self, X):
        """
        Apply a sequence of physical transformations to input data.

        The transformations are applied in the following order:
            1. Canonicalization of geometries (if enabled)
            2. Morse transformation (if enabled)
            3. Permutationally invariant polynomial (PIP) transformation (if enabled)

        Args:
            X (np.ndarray): Input feature array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed feature array.
        """
        # 1) canonicalize geometries
        if self.cfg["canonicalize_geom"]:
            X = canonicalize_geom(X, molecule=self.cfg["molecule"])
        # 2) apply morse transform
        if self.cfg["morse_transform"]:
            X = morse(X, morse_param=self.cfg["morse_param"])
        # 3) apply pip transform
        if self.cfg["pip_transform"]:
            X = pips(X, molecule=self.cfg["molecule"])
        return X

    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline to the data.

        Applies physical transformations to `X`, then fits the feature scaler.
        Optionally fits a separate scaler for target values `y`.

        Args:
            X (np.ndarray): Input feature array of shape (n_samples, n_features).
            y (np.ndarray, optional): Target array. Defaults to None.

        Returns:
            Preprocessor: Fitted preprocessor instance.
        """
        X = self._physical_transform(X)
        self.scaler_X = self._create_scaler(self.cfg["scaler_data"])
        self.scaler_X.fit(X)
        logger.info("Fitted data.")

        if y is not None:
            self.scaler_y = self._create_scaler(self.cfg["scaler_target"])
            self.scaler_y.fit(y)
            self.fitted_y = True
            logger.info("Fitted target.")

        self.fitted_X = True

        return self

    def transform(self, X, y=None):
        """
        Transform input data using the fitted preprocessing pipeline.

        Applies physical transformations followed by scaling. If `y` is provided,
        it is also scaled using the fitted target scaler.

        Args:
            X (np.ndarray): Input feature array of shape (n_samples, n_features).
            y (np.ndarray, optional): Target array. Defaults to None.

        Returns:
            np.ndarray or tuple[np.ndarray, np.ndarray]:
                - Transformed features if `y` is None
                - Tuple of transformed features and targets if `y` is provided
        """
        if not self.fitted_X:
            raise RuntimeError("Data scaler not fitted before transform().")

        X = self._physical_transform(X)
        X = self.scaler_X.transform(X)

        if y is not None:
            if not self.fitted_y:
                raise RuntimeError("Target scaler not fitted before transform().")
            y = self.scaler_y.transform(y)
            return X, y

        return X

    def inverse_transform_y(self, y):
        """
        Inverse transform scaled target values.

        Args:
            y (np.ndarray): Scaled target array.

        Returns:
            np.ndarray: Original-scale target values.
        """
        if not self.fitted_y:
            raise RuntimeError("Target scaler not fitted before inverse transform.")
        return self.scaler_y.inverse_transform(y)

    def output_dim(self, X):
        """
        Compute the output feature dimension after preprocessing.

        Applies the physical transformations to a single sample to determine
        the resulting number of features.

        Args:
            X (np.ndarray): Input feature array.

        Returns:
            int: Number of output features after transformation.
        """
        Xp = self._physical_transform(X[:1])
        return Xp.shape[1]
