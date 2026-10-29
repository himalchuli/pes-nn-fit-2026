import logging
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from pes_nn_fit.utils.helpers import (
    au2angstrom,
    hartree2cm1,
    hartree2ev,
    interatom2zmatrix,
    zmatrix2interatom,
)

logger = logging.getLogger(__name__)


class Dataset:
    """
    Handles data extraction, unit conversion, coordinate transformation and splitting.
    """

    def __init__(
        self,
        file_location: str = None,
        file_type: str = "pes-learn",
        array: np.ndarray = None,
        array_tuple: np.ndarray = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize the dataset.

        Multiple methods of creating dataset can be used:
        - load from a file
        - from array [X,y]
        - from array_tuple (X,y)

        Parameters
        ----------
        file_location : str, optional
            Path to a CSV file where all columns except the last are features
            and the last column is the target variable.
            Typically ``config['data_file']``.
            Must provide file_type.
        file_type : str,optional
            Controls the importing scheme of file in file_location. All use
            np.loadtxt with different setting. Options:

            -`'pes-learn'` (default): delimiter=",", skiprows=1

            -`'molpro'`: file_location, usecols=(0, 1, 2, 4)

            -`'psi4'`: delimiter=",", skiprows=1, usecols=(3, 4, 5, 6)

        array : numpy.ndarray, optional
            Input data provided directly as an array of which all columns except
            the last are features and the last column are targets.

        array_tuple : tuple[numpy.ndarray, numpy.ndarray], optional
            First array is data, second array is target.

        seed : int, optional
            Random seed for reproducibility.
        """

        if file_location is not None:
            if file_type == "pes-learn":
                self.dataset = np.loadtxt(file_location, delimiter=",", skiprows=1)
            elif file_type == "psi4":
                self.dataset = np.loadtxt(
                    file_location, delimiter=",", skiprows=1, usecols=(3, 4, 5, 6)
                )
            elif file_type == "molpro":
                self.dataset = np.loadtxt(file_location, usecols=(0, 1, 2, 4))
            elif file_type == "random_dirs":
                self.dataset = np.loadtxt(file_location, usecols=(0, 1, 3, 5))
            else:
                raise ValueError("File type %s not supported.")

        elif array is not None:
            self.dataset = array
        elif array_tuple is not None:
            self.dataset = np.column_stack(array_tuple)
        else:
            raise ValueError("Provide either file location or an array")
        self.data = self.dataset[:, :-1]
        self.target = self.dataset[:, -1].reshape(-1, 1)
        self.seed = seed
        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]
        logger.debug(
            "Dataset initialized. Shape: (%d, %d)", self.n_samples, self.n_features
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def __str__(self):
        return str(np.column_stack((self.data, self.target)))

    def combine(self, a):
        """
        Combine the current Dataset with another Dataset a.
        """
        data = np.vstack((self.data, a.data))
        target = np.vstack((self.target, a.target))
        return Dataset(array_tuple=(data, target))

    def save(self, path):
        """Save dataset to `path`."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.dataset)
        logger.info("Dataset saved at %s", path)
        # TODO: Currently Seed is lost during saving, but I don't think I need it for now

    @classmethod
    def load(cls, path):
        """Load dataset from `path`."""
        loaded_dataset = np.load(path)
        dataset = cls(array=loaded_dataset)
        logger.info("Dataset loaded from %s", path)
        return dataset

    def mirror(self, molecule: str) -> "Dataset":
        """
        Augment the dataset by applying symmetry-based mirroring operations.

        For supported molecules, symmetry-equivalent configurations are generated
        and added to the dataset. This is useful when the original dataset contains
        only symmetry-unique points.

        Currently supported:
        - ``'h2o'``: Swaps the two O–H bond lengths (assumes internal coordinates
        ordered as [rOH1, rOH2, angle]). The mirrored configurations are appended
        and duplicate rows are removed.
        - ``'n2o'``: No symmetry expansion is applied.

        Parameters
        ----------
        molecule : str
            Identifier of the molecule. Supported values are:
            ``'h2o'``, ``'n2o'``.

        Returns
        -------
        Dataset
            The current Dataset instance with symmetry-expanded data.

        Notes
        -----
        - The operation is performed in-place: ``self.dataset``, ``self.data``,
        ``self.target``, and shape attributes are updated.
        - Duplicate rows introduced by mirroring are removed using
        ``np.unique(..., axis=0)``.
        - For ``'h2o'``, input data must already be in internal coordinates
        (rOH1, rOH2, angle).
        """

        if molecule == "h2o":
            # water must be in internal coords (rOH, rOH, aHOH)
            mirror = self.dataset.copy()
            mirror[:, 1] = self.data[:, 0]
            mirror[:, 0] = self.data[:, 1]
            self.dataset = np.unique(np.vstack([self.dataset, mirror]), axis=0)
            self.data = self.dataset[:, :-1]
            self.target = self.dataset[:, -1].reshape(-1, 1)
            self.n_samples = self.data.shape[0]
            self.n_features = self.data.shape[1]
        elif molecule == "n2o":
            # nothing to mirror
            pass
        else:
            raise ValueError(f"Molecule {molecule} not supported.")
        return self

    def interatom2zmatrix(self, molecule: str) -> "Dataset":
        """
        Convert interatomic distance representation to Z-matrix coordinates.

        Uses ``pes_nn_fit.utils.helpers.interatom2zmatrix`` to perform the transformation
        from interatomic distances to internal (Z-matrix) coordinates.

        Parameters
        ----------
        molecule : str
            Molecule identifier passed to the transformation function.

        Returns
        -------
        Dataset
            The updated dataset (in-place).
        """
        self.data = interatom2zmatrix(self.data, molecule)
        self.dataset[:, :-1] = self.data
        return self

    def zmatrix2interatom(self, molecule):
        """
        Convert Z-matrix coordinates to interatomic distance representation.

        Uses `pes_nn_fit.utils.helpers.zmatrix2interatom` to transform internal (Z-matrix) coordinates
        into interatomic distances.

        Parameters
        ----------
        molecule : str
            Molecule identifier passed to the transformation function.

        Returns
        -------
        Dataset
            The updated dataset (in-place).
        """
        self.data = zmatrix2interatom(self.data, molecule)
        self.dataset[:, :-1] = self.data
        return self

    def energy_hartree2cm1(self) -> "Dataset":
        """
        Convert energy values from Hartree to inverse centimeters (cm⁻¹).

        Uses `pes_nn_fit.utils.helpers.hartree2cm1` to transform the energy column of the dataset
        and updates the target accordingly.

        Returns
        -------
        Dataset
            The updated dataset (in-place).
        """
        self.dataset[:, -1] = hartree2cm1(self.dataset[:, -1])
        self.target = self.dataset[:, -1].reshape(-1, 1)
        return self

    def energy_hartree2ev(self) -> "Dataset":
        """
        Convert energy values from Hartree to electronvolts (eV).

        Uses `pes_nn_fit.utils.helpers.hartree2ev` to transform the energy column of the dataset
        and updates the target accordingly.

        Returns
        -------
        Dataset
            The updated dataset (in-place).
        """
        self.dataset[:, -1] = hartree2ev(self.dataset[:, -1])
        self.target = self.dataset[:, -1].reshape(-1, 1)
        return self

    def dists_au2angstrom(self) -> "Dataset":
        """
        Convert interatomic distances from atomic units (Bohr) to Angstrom.

        Uses `pes_nn_fit.utils.helpers.au2angstrom` to transform distance values and updates
        the dataset accordingly.

        Returns
        -------
        Dataset
            The updated dataset (in-place).
        """
        self.data = au2angstrom(self.data)
        self.dataset[:, :-1] = self.data
        return self

    def energy_subtract_min(self, E_min) -> "Dataset":
        """Shift energy values of the dataset by subtracting `E_min`."""
        self.dataset[:, -1] = self.dataset[:, -1] - E_min
        self.target = self.dataset[:, -1].reshape(-1, 1)
        return self

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_split_sizes(train_size, val_size, test_size) -> float:
        """
        Validate relative split sizes.

        ``val_size=None`` is treated as ``0`` (no validation set).

        Returns
        -------
        float
            The validated ``val_size`` (``0.0`` if it was ``None``).

        Raises
        ------
        ValueError
            If a size is outside [0, 1], if ``train_size`` or ``test_size`` is
            not positive, or if the three sizes do not sum to 1.
        """
        if val_size is None:
            val_size = 0.0

        for name, value in (
            ("train_size", train_size),
            ("val_size", val_size),
            ("test_size", test_size),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the interval [0, 1], got {value}.")
        if train_size <= 0.0:
            raise ValueError("train_size must be greater than 0.")
        if test_size <= 0.0:
            raise ValueError("test_size must be greater than 0.")

        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"train_size + val_size + test_size must sum to 1, got {total} "
                f"({train_size} + {val_size} + {test_size})."
            )
        return float(val_size)

    @staticmethod
    def _split_array(array, val_size, test_size, seed):
        """
        Randomly split ``array`` into (train, val, test) or, if ``val_size == 0``,
        into (train, test).
        """
        if val_size == 0:
            return tuple(train_test_split(array, test_size=test_size, random_state=seed))
        trainset, helpset = train_test_split(
            array, test_size=val_size + test_size, random_state=seed
        )
        valset, testset = train_test_split(
            helpset, test_size=test_size / (val_size + test_size), random_state=seed
        )
        return trainset, valset, testset

    def _wrap(self, parts):
        """Wrap each array in ``parts`` into a Dataset."""
        return tuple(Dataset(array=part, seed=self.seed) for part in parts)

    @staticmethod
    def _log_split(name, parts):
        labels = ("Train", "Val", "Test") if len(parts) == 3 else ("Train", "Test")
        sizes = " | ".join(f"{label}: {len(part)}" for label, part in zip(labels, parts))
        logger.info("%s split completed. %s", name, sizes)

    def split_random(self, train_size=0.8, val_size=0.1, test_size=0.1):
        """
        Split the dataset into train/validation/test subsets using a random split.

        Parameters
        ----------
        train_size : float, default=0.8
            Relative size of the training set. Must be in (0, 1].

        val_size : float or None, default=0.1
            Relative size of the validation set. Must be in [0, 1].
            If ``None`` or ``0``, no validation set is created.

        test_size : float, default=0.1
            Relative size of the test set. Must be in (0, 1].

        Returns
        -------
        tuple of Dataset
            ``(trainset, valset, testset)``, or ``(trainset, testset)`` if
            ``val_size`` is ``None`` or ``0``.

        Raises
        ------
        ValueError
            If ``train_size + val_size + test_size`` does not equal 1.

        Notes
        -----
        Splitting is performed using ``sklearn.model_selection.train_test_split``.
        """
        val_size = self._validate_split_sizes(train_size, val_size, test_size)
        parts = self._split_array(self.dataset, val_size, test_size, self.seed)
        self._log_split("Random", parts)
        return self._wrap(parts)

    def split_quantiles(
        self, train_size=0.8, val_size=0.1, test_size=0.1, n_quantiles=10
    ):
        """
        Split the dataset into subsets using target quantiles.

        The samples are first sorted by the target value and divided into
        ``n_quantiles`` equally sized bins. Each bin is then split independently,
        which approximately preserves the target distribution across the final
        subsets.

        Parameters
        ----------
        train_size : float, default=0.8
            Relative size of the training set. Must be in (0, 1].

        val_size : float or None, default=0.1
            Relative size of the validation set. Must be in [0, 1].
            If ``None`` or ``0``, no validation set is created.

        test_size : float, default=0.1
            Relative size of the test set. Must be in (0, 1].

        n_quantiles : int, default=10
            Number of target quantile bins used before splitting.

        Returns
        -------
        tuple of Dataset
            ``(trainset, valset, testset)``, or ``(trainset, testset)`` if
            ``val_size`` is ``None`` or ``0``.

        Raises
        ------
        ValueError
            If ``train_size + val_size + test_size`` does not equal 1.

        Notes
        -----
        Splitting is performed independently inside each quantile using
        ``sklearn.model_selection.train_test_split``. A deterministic but
        varying random seed is used per quantile to maintain reproducibility
        while avoiding identical partitions.
        """
        val_size = self._validate_split_sizes(train_size, val_size, test_size)

        idx_sort = self.target.ravel().argsort()
        dataset = np.column_stack((self.data, self.target))[idx_sort]
        bins = np.array_split(dataset, n_quantiles)

        per_bin = []
        seed = self.seed
        for bin_ in bins:
            per_bin.append(self._split_array(bin_, val_size, test_size, seed))
            seed += 1  # different seed for every quantile but still replicable

        parts = tuple(np.vstack(group) for group in zip(*per_bin))
        self._log_split("Quantile", parts)
        return self._wrap(parts)

    def split_fps(self, train_size=0.8, val_size=0.1, test_size=0.1):
        """
        Split the dataset using farthest point sampling (FPS).

        The training subset is constructed iteratively. Starting from the sample
        with the lowest target value (interpreted as the equilibrium point),
        each subsequent sample is chosen as the one whose minimum distance to
        all previously selected training samples is maximal. This produces a
        diverse training set spanning the feature space. The remaining samples
        are split into validation and test subsets.

        Parameters
        ----------
        train_size : float, default=0.8
            Relative size of the training set. Must be in (0, 1].

        val_size : float or None, default=0.1
            Relative size of the validation set. Must be in [0, 1].
            If ``None`` or ``0``, all remaining samples form the test set.

        test_size : float, default=0.1
            Relative size of the test set. Must be in (0, 1].

        Returns
        -------
        tuple of Dataset
            ``(trainset, valset, testset)``, or ``(trainset, testset)`` if
            ``val_size`` is ``None`` or ``0``.

        Raises
        ------
        ValueError
            If ``train_size + val_size + test_size`` does not equal 1.

        Notes
        -----
        Distances are computed in feature space using Euclidean norm.
        After constructing the training subset, the remaining samples are split
        with ``sklearn.model_selection.train_test_split`` using ``self.seed`` for
        reproducibility. This strategy produces a well-spread training set but
        typically extrapolates poorly outside the sampled data domain.

        References
        ----------
        Dral, P. O.; Owens, A.; Yurchenko, S. N.; Thiel, W.
        Structure-based sampling and self-correcting machine learning for
        accurate calculations of potential energy surfaces and vibrational
        levels. *J. Chem. Phys.* 2017, 146, 244108.
        """
        val_size = self._validate_split_sizes(train_size, val_size, test_size)

        dataset = self.dataset
        X = self.data
        y = self.target

        def distances(point):
            """Computes norm between training point geometry and every point in dataset"""
            diff = X - point
            return np.sqrt(np.sum(diff * diff, axis=1))

        train_idxs = []
        # select the point with lowest energy as starting point
        idx_min = y.ravel().argsort()[0]
        train_idxs.append(idx_min)
        first_dists = distances(X[idx_min])
        # find the farthest point from the 1st point
        idx_far = np.argmax(first_dists)
        train_idxs.append(idx_far)

        min_dists = np.minimum(first_dists, distances(X[idx_far]))

        n_train = int(len(dataset) * train_size)
        for _ in range(2, n_train):
            idx_new = np.argmax(min_dists)
            train_idxs.append(idx_new)

            new_dists = distances(X[idx_new])
            min_dists = np.minimum(min_dists, new_dists)

        train_idxs = np.asarray(train_idxs)
        trainset = dataset[train_idxs]
        helpset = np.delete(dataset, train_idxs, axis=0)

        if val_size == 0:
            parts = (trainset, helpset)
        else:
            valset, testset = train_test_split(
                helpset,
                test_size=test_size / (test_size + val_size),
                random_state=self.seed,
            )
            parts = (trainset, valset, testset)

        self._log_split("FPS", parts)
        return self._wrap(parts)

    def split_kmeans(
        self, train_size=0.8, val_size=0.1, test_size=0.1, n_clusters="default"
    ):
        """
        Split the dataset using k-means clustering.

        Samples are first partitioned into ``n_clusters`` groups in feature space.
        Each cluster is then split independently into training, validation, and
        test subsets. This approximately preserves coverage of different regions
        of the feature space across all subsets.

        Parameters
        ----------
        train_size : float, default=0.8
            Relative size of the training set. Must be in (0, 1].

        val_size : float or None, default=0.1
            Relative size of the validation set. Must be in [0, 1].
            If ``None`` or ``0``, no validation set is created.

        test_size : float, default=0.1
            Relative size of the test set. Must be in (0, 1].

        n_clusters : int or 'default', default='default'
            Number of clusters used for k-means partitioning. If ``'default'``,
            the number of clusters is set to ``max(2, len(self) // 30)``.

        Returns
        -------
        tuple of Dataset
            ``(trainset, valset, testset)``, or ``(trainset, testset)`` if
            ``val_size`` is ``None`` or ``0``.

        Raises
        ------
        ValueError
            If ``train_size + val_size + test_size`` does not equal 1.

        Notes
        -----
        Clustering is performed using ``sklearn.cluster.KMeans`` with
        ``random_state=self.seed`` for reproducibility. Splitting inside each
        cluster uses ``sklearn.model_selection.train_test_split``. Clusters
        containing fewer than three samples are assigned entirely to the
        training subset to avoid unstable splits.
        """
        val_size = self._validate_split_sizes(train_size, val_size, test_size)

        if n_clusters == "default":
            n_clusters = max(2, len(self) // 30)
        # later can be clustering also applied to energies but for now only for params
        dataset = self.dataset
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.seed)
        labels = kmeans.fit_predict(self.data)

        n_parts = 2 if val_size == 0 else 3
        empty = np.empty((0, dataset.shape[1]))

        per_cluster = []
        for cluster_label in range(n_clusters):
            idx_label = np.where(labels == cluster_label)[0]
            # skip if there aren't any points in the cluster
            if len(idx_label) == 0:
                continue
            cluster = np.atleast_2d(dataset[idx_label])
            # If cluster too small -> put all in train
            if len(idx_label) < 3:
                per_cluster.append((cluster,) + (empty,) * (n_parts - 1))
                continue
            per_cluster.append(
                self._split_array(cluster, val_size, test_size, self.seed)
            )

        parts = tuple(np.vstack(group) for group in zip(*per_cluster))
        self._log_split("K-means clustering", parts)
        return self._wrap(parts)

    def split(
        self,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        split_type="random",
    ):
        """
        Split the dataset using the chosen strategy.

        Parameters
        ----------
        train_size, val_size, test_size : float
            Relative sizes of the subsets. They must sum to 1, otherwise a
            ``ValueError`` is raised. If ``val_size`` is ``None`` or ``0``,
            only train and test sets are returned.

        split_type : {'random', 'quantiles', 'fps', 'kmeans'}, default='random'
            Splitting strategy.

        Returns
        -------
        tuple of Dataset
            ``(trainset, valset, testset)``, or ``(trainset, testset)`` if
            ``val_size`` is ``None`` or ``0``.
        """
        split_methods = {
            "random": self.split_random,
            "quantiles": self.split_quantiles,
            "fps": self.split_fps,
            "kmeans": self.split_kmeans,
        }

        if split_type not in split_methods:
            raise ValueError(f"Invalid split_type: {split_type}")
        return split_methods[split_type](
            train_size=train_size, val_size=val_size, test_size=test_size
        )
