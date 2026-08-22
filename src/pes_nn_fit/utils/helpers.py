import numpy as np
from typing import Union

# region Config


def merge_config(base: dict, override: dict) -> dict:
    """
    Recursively merge two configuration dictionaries.

    Values from ``override`` take precedence over ``base``. Nested dictionaries
    are merged recursively, except for nodes containing a ``"dist"`` key, which
    are treated as atomic and replaced entirely.

    Typically used to merge `default.yaml` config with the user defined one.

    Parameters
    ----------
    base : dict
        Base configuration dictionary.
    override : dict
        Configuration dictionary whose values override those in ``base``.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """

    if not isinstance(base, dict) or not isinstance(override, dict):
        return override

    # distribution nodes are atomic
    if "dist" in base or "dist" in override:
        return override

    merged = dict(base)

    for k, v in override.items():
        if k in base:
            merged[k] = merge_config(base[k], v)
        else:
            merged[k] = v

    return merged


def sample_config(trial, node, prefix: str = ""):
    """
    Sample a configuration from a hierarchical search space definition.

    Traverses a nested configuration structure and samples values for nodes
    that define a distribution via a ``"dist"`` key. Supports float, int, and
    categorical distributions using the provided ``trial`` object.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object used for sampling.
    node : dict or any
        Configuration node. Can be:
        - dict with ``"dist"`` key (distribution definition),
        - nested dict (recursively processed),
        - constant value (returned as-is).
    prefix : str, optional
        Parameter name prefix used to construct unique names for sampling.

    Returns
    -------
    dict or any
        Sampled configuration with the same structure as ``node``.
    """

    # distribution node
    if isinstance(node, dict) and "dist" in node:

        name = prefix
        dist = node["dist"]

        if dist == "float":
            return trial.suggest_float(
                name, node["low"], node["high"], log=node.get("log", False)
            )

        elif dist == "int":
            return trial.suggest_int(name, node["low"], node["high"])

        elif dist == "categorical":
            return trial.suggest_categorical(name, node["choices"])

        else:
            raise ValueError(f"Unknown dist {dist}")

    # recursive dict
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            out[k] = sample_config(trial, v, new_prefix)
        return out

    # constant leaf
    return node


def unflatten_params(flat: dict) -> dict:
    """Convert {'a.b.c': v} → {'a': {'b': {'c': v}}}"""
    nested = {}
    for key, value in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested


# endregion


# region Coordinate Transformations


def zmatrix2interatom(zmatrix: np.ndarray, molecule: str) -> np.ndarray:
    """
    Convert Z-matrix internal coordinates to interatomic distances.

    Supports specific molecules with predefined coordinate conventions.

    Parameters
    ----------
    zmatrix : ndarray, shape (N, 3) or (3,)
        Internal coordinates:
        - h2o: (rOH₁, rOH₂, ∠HOH) with angle in degrees
        - n2o: (rN₁N₂, rN₂O, ∠N₁ON₂)
    molecule : str
        Molecule identifier (e.g., "h2o", "n2o").

    Returns
    -------
    ndarray, shape (N, 3)
        Interatomic distances:
        - h2o: (rHH, rOH₁, rOH₂)
        - n2o: (rN₁N₂, rN₂O, rN₁O)
    """

    # Force shape (n,3)
    if zmatrix.ndim == 1:
        zmatrix = zmatrix.reshape(1, -1)

    interatom = np.zeros_like(zmatrix)

    r1 = zmatrix[:, 0]
    r2 = zmatrix[:, 1]
    theta = zmatrix[:, 2]

    theta_rad = np.deg2rad(theta)

    if molecule == "h2o":
        rHH = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(theta_rad))
        # Order: (rHH, rOH, rOH)
        interatom[:, 0] = rHH
        interatom[:, 1] = r1
        interatom[:, 2] = r2

    if molecule == "n2o":
        rN1O = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(theta_rad))
        interatom[:, 0] = r1
        interatom[:, 1] = r2
        interatom[:, 2] = rN1O
    return interatom


def interatom2zmatrix(interatom: np.ndarray, molecule: str) -> np.ndarray:
    """
    Convert interatomic distances to Z-matrix internal coordinates.

    Parameters
    ----------
    interatom : ndarray, shape (N, 3) or (3,)
        Interatomic distances:
        - h2o: (rHH, rOH₁, rOH₂)
        - n2o: (rN₁N₂, rN₂O, rN₁O)
    molecule : str
        Molecule identifier (e.g., "h2o", "n2o").

    Returns
    -------
    ndarray, shape (N, 3)
        Z-matrix internal coordinates:
        - h2o: (rOH₁, rOH₂, ∠HOH) with angle in degrees
        - n2o: (rN₁N₂, rN₂O, ∠N₁ON₂)

        Angles are returned in degrees.
    """
    if interatom.ndim == 1:
        interatom = interatom.reshape(1, -1)

    zmatrix = np.zeros_like(interatom)
    r0, r1, r2 = [interatom[:, i] for i in range(3)]

    if molecule == "h2o":
        # r0=rHH, r1=rOH1, r2=rOH2 → angle at O
        cos = (r2**2 + r1**2 - r0**2) / (2 * r1 * r2)
        theta = np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))
        zmatrix[:, [0, 1]] = interatom[:, [1, 2]]
        zmatrix[:, 2] = theta

    elif molecule == "n2o":
        # r0=rN1N2, r1=rN2O, r2=rN1O → angle at N2
        cos = (r0**2 + r1**2 - r2**2) / (2 * r0 * r1)  # ← was using wrong formula
        theta = np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))
        zmatrix[:, [0, 1]] = interatom[:, [0, 1]]
        zmatrix[:, 2] = theta

    return zmatrix


# endregion

# region Unit Conversion


def hartree2cm1(energy_hartree: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Convert energy from Hartree to wavenumbers (cm⁻¹)."""
    HARTREE_TO_CM1 = 219474.631363
    return energy_hartree * HARTREE_TO_CM1


def hartree2ev(energy_hartree: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Convert energy from Hartree to electronvolts."""
    HARTREE_TO_eV = 27.2114079527
    return HARTREE_TO_eV * energy_hartree


def au2angstrom(dist_au: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Convert distance from atomic units to Angstrom.."""
    AU_TO_ANGSTROM = 0.529177
    return AU_TO_ANGSTROM * dist_au


# endregion


class GP_wrapper:
    """Wrapper for a Gaussian Process model with input/output standardization."""

    def __init__(self, gp, stdX, stdy):
        self.gp = gp
        self.stdX = stdX
        self.stdy = stdy

    def __call__(self, x):
        x = self.stdX.transform(x)
        y = self.gp.predict(x)
        E_hartree = self.stdy.inverse_transform(y.reshape(-1, 1))
        return E_hartree
