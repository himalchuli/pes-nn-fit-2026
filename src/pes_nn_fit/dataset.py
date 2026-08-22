import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import logging
from src.utils.helpers import hartree2cm1, hartree2ev, au2angstrom
from src.utils.helpers import interatom2zmatrix, zmatrix2interatom

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

        Uses ``src.utils.helpers.interatom2zmatrix`` to perform the transformation
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

        Uses `src.utils.helpers.zmatrix2interatom` to transform internal (Z-matrix) coordinates
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

        Uses `src.utils.helpers.hartree2cm1` to transform the energy column of the dataset
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

        Uses `src.utils.helpers.hartree2ev` to transform the energy column of the dataset
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

        Uses `src.utils.helpers.au2angstrom` to transform distance values and updates
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

    def split_random(self, val_size=0.1, test_size=0.1):
        """
        Splits the dataset into train set, validation set and test set using random split.

        Parameters
        ----------
        val_size : float
            Relative size of the validation set. Must be in the interval [0, 1].

        test_size : float
            Relative size of the test set. Must be in the interval [0, 1].

        Returns
        -------
        trainset : Dataset
            Training subset.

        valset : Dataset
            Validation subset.

        testset : Dataset
            Test subset.

        Notes
        -----
        Splitting is performed using ``sklearn.model_selection.train_test_split``.
        """

        trainset, helpset = train_test_split(
            self.dataset, test_size=test_size + val_size, random_state=self.seed
        )
        valset, testset = train_test_split(
            helpset,
            test_size=test_size / (test_size + val_size),
            random_state=self.seed,
        )

        logger.info(
            "Random split completed. Train: %d | Val: %d | Test: %d",
            len(trainset),
            len(valset),
            len(testset),
        )

        return (
            Dataset(array=trainset, seed=self.seed),
            Dataset(array=valset, seed=self.seed),
            Dataset(array=testset, seed=self.seed),
        )

    def split_quantiles(self, val_size=0.1, test_size=0.1, n_quantiles=10):
        """
        Split the dataset into train/validation/test subsets using target quantiles.

        The samples are first sorted by the target value and divided into
        ``n_quantiles`` equally sized bins. Each bin is then split independently,
        which approximately preserves the target distribution across the final
        subsets.

        Parameters
        ----------
        val_size : float, default=0.1
            Relative size of the validation subset within the full dataset.
            Must be in the interval [0, 1].

        test_size : float, default=0.1
            Relative size of the test subset within the full dataset.
            Must be in the interval [0, 1].

        n_quantiles : int, default=10
            Number of target quantile bins used before splitting.

        Returns
        -------
        trainset : Dataset
            Training subset assembled from all quantile splits.

        valset : Dataset
            Validation subset assembled from all quantile splits.

        testset : Dataset
            Test subset assembled from all quantile splits.

        Notes
        -----
        Splitting is performed independently inside each quantile using
        ``sklearn.model_selection.train_test_split``. A deterministic but
        varying random seed is used per quantile to maintain reproducibility
        while avoiding identical partitions.
        """

        X = self.data
        y = self.target
        idx_sort = y.ravel().argsort()
        dataset = np.column_stack((X, y))[idx_sort]
        splits = np.array_split(dataset, n_quantiles)

        trainset = []
        valset = []
        testset = []
        seed = self.seed
        for split in splits:
            trainset_split, helpset = train_test_split(
                split, test_size=test_size + val_size, random_state=seed
            )
            valset_split, testset_split = train_test_split(
                helpset, test_size=test_size / (test_size + val_size), random_state=seed
            )
            trainset.append(trainset_split)
            valset.append(valset_split)
            testset.append(testset_split)
            seed += 1  # we want use different seed for every quantile but still have replicable results

        logger.info(
            "Quantile split completed. Train: %d | Val: %d | Test: %d",
            len(trainset),
            len(valset),
            len(testset),
        )

        return (
            Dataset(array=np.vstack(trainset), seed=self.seed),
            Dataset(array=np.vstack(valset), seed=self.seed),
            Dataset(array=np.vstack(testset), seed=self.seed),
        )

    def split_fps(self, val_size: float = 0.1, test_size: float = 0.1) -> float:
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
        val_size : float, default=0.1
            Relative size of the validation subset within the full dataset.
            Must be in the interval [0, 1].

        test_size : float, default=0.1
            Relative size of the test subset within the full dataset.
            Must be in the interval [0, 1].

        Returns
        -------
        trainset : Dataset
            Training subset selected using farthest point sampling.

        valset : Dataset
            Validation subset sampled from the remaining data.

        testset : Dataset
            Test subset sampled from the remaining data.

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

        n_train = int(len(dataset) * (1 - (val_size + test_size)))
        for i in range(2, n_train):
            idx_new = np.argmax(min_dists)
            train_idxs.append(idx_new)

            new_dists = distances(X[idx_new])
            min_dists = np.minimum(min_dists, new_dists)

        train_idxs = np.asarray(train_idxs)
        trainset = dataset[train_idxs]
        helpset = np.delete(dataset, train_idxs, axis=0)
        valset, testset = train_test_split(
            helpset,
            test_size=test_size / (test_size + val_size),
            random_state=self.seed,
        )

        logger.info(
            "FPS split completed. Train: %d | Val: %d | Test: %d",
            len(trainset),
            len(valset),
            len(testset),
        )

        return (
            Dataset(array=trainset, seed=self.seed),
            Dataset(array=valset, seed=self.seed),
            Dataset(array=testset, seed=self.seed),
        )

    def split_kmeans(self, val_size=0.1, test_size=0.1, n_clusters="default"):
        """
        Split the dataset using k-means clustering.

        Samples are first partitioned into ``n_clusters`` groups in feature space.
        Each cluster is then split independently into training, validation, and
        test subsets. This approximately preserves coverage of different regions
        of the feature space across all subsets.

        Parameters
        ----------
        val_size : float, default=0.1
            Relative size of the validation subset within the full dataset.
            Must be in the interval [0, 1].

        test_size : float, default=0.1
            Relative size of the test subset within the full dataset.
            Must be in the interval [0, 1].

        n_clusters : int or 'default', default='default'
            Number of clusters used for k-means partitioning. If ``'default'``,
            the number of clusters is set to ``max(2, len(self) // 30)``.

        Returns
        -------
        trainset : Dataset
            Training subset assembled from all cluster splits.

        valset : Dataset
            Validation subset assembled from all cluster splits.

        testset : Dataset
            Test subset assembled from all cluster splits.

        Notes
        -----
        Clustering is performed using ``sklearn.cluster.KMeans`` with
        ``random_state=self.seed`` for reproducibility. Splitting inside each
        cluster uses ``sklearn.model_selection.train_test_split``. Clusters
        containing fewer than three samples are assigned entirely to the
        training subset to avoid unstable splits.
        """

        if n_clusters == "default":
            n_clusters = max(2, len(self) // 30)  # recommended by chatgpt
        # later can be clustering also applied to energies but for now only for params
        X = self.data
        dataset = self.dataset
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.seed)
        labels = kmeans.fit_predict(X)

        trainset = []
        valset = []
        testset = []
        for cluster_label in range(n_clusters):
            idx_label = np.where(labels == cluster_label)[0]
            # skip if there aren't any points in the cluster
            if len(idx_label) == 0:
                continue
            cluster = np.atleast_2d(
                dataset[idx_label]
            )  # ensures 1d array will also vstack
            # If cluster too small → put all in train
            if len(idx_label) < 3:
                trainset.append(cluster)
                continue
            trainset_split, helpset = train_test_split(
                cluster, test_size=test_size + val_size, random_state=self.seed
            )
            valset_split, testset_split = train_test_split(
                helpset,
                test_size=test_size / (test_size + val_size),
                random_state=self.seed,
            )
            trainset.append(trainset_split)
            valset.append(valset_split)
            testset.append(testset_split)

        logger.info(
            "K-means clustering split completed. Train: %d | Val: %d | Test: %d",
            len(trainset),
            len(valset),
            len(testset),
        )

        return (
            Dataset(array=np.vstack(trainset), seed=self.seed),
            Dataset(array=np.vstack(valset), seed=self.seed),
            Dataset(array=np.vstack(testset), seed=self.seed),
        )

    def split(self, val_size=0.1, test_size=0.1, split_type="random"):
        split_methods = {
            "random": self.split_random,
            "quantiles": self.split_quantiles,
            "fps": self.split_fps,
            "kmeans": self.split_kmeans,
        }

        try:
            return split_methods[split_type](val_size, test_size)
        except KeyError:
            raise ValueError(f"Invalid split_type: {split_type}")
