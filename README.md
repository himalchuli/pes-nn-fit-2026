# pes-nn-fit-2026: Neural Network Fitting for Potential Energy Surfaces

A Python and PyTorch package for fitting high-precision potential energy surfaces (PES) of small molecules using Permutationally Invariant Polynomial Neural Networks (PIP-NN).

This repository contains the implementation developed for the master's thesis *Using neural networks for calculating potential energy surfaces of molecules* (Charles University, 2026).

## What It's For

Constructing accurate potential energy surfaces from discrete *ab initio* quantum chemistry calculations (such as Molpro or PSI4) is essential for molecular dynamics and vibrational spectroscopy. Standard interpolation methods like splines suffer from the curse of dimensionality, while empirical potentials lack precision. 

This framework solves this by combining:
1. **Permutationally Invariant Polynomials (PIPs)** and Morse variable transforms to enforce exchange symmetries of indistinguishable nuclei and correct asymptotic decay.
2. **Feed-Forward Neural Networks** trained with the second-order **Levenberg–Marquardt (L-M)** algorithm, achieving near-spectral accuracy on small molecular datasets where standard first-order optimizers (like Adam) fall short.

## Key Features

- **Geometry Preprocessing:** Coordinate conversions (Z-matrix ↔ interatomic distances), geometry canonicalization, Morse transformations (e^(-ar)), and molecular PIP mappings. Currently pre-configured for H₂O and N₂O.

- **Sampling & Splitting Strategies:**
  - Random split.
  - Stratified energy quantile split.
  - Spatial *k*-means clustering.
  - Farthest Point Sampling (FPS) for maximal feature-space coverage.

- **High-Precision Training:** PyTorch models running at `float64` precision utilizing the Levenberg–Marquardt optimization algorithm (or sequential Adam → L-M) with early stopping and checkpointing.

- **Automated Hyperparameter Tuning:** Automated search via Optuna (TPE sampling, median pruning) for layer sizing, learning rates, and Morse decay parameters.

- **End-to-End Pipeline:** Unified `Pipeline` class wrapping data scaling, model architecture, training, persistence, and inverse-transformed predictions.

## Installation

### Prerequisites

A Python 3.10 environment is recommended.

### Install from Source

```
git clone [https://github.com/himalchuli/pes-nn-fit-2026.git](https://github.com/himalchuli/pes-nn-fit-2026.git)
cd pes-nn-fit-2026
pip install .
```

## File Structure
```
├── examples/               # Demonstration notebooks (water fitting, hyperparameter pipelines)
├── src/
│   └── pes_nn_fit/
│       ├── dataset.py      # Dataset loading, coordinate transformations, and splitting routines
│       ├── hypersearch.py  # Optuna optimization wrapper
│       ├── model.py        # PyTorch NN architecture, Trainer (Adam/LM), and Pipeline class
│       ├── preprocess.py   # Canonicalization, Morse transform, PIP transform, and scalers
│       └── utils/          # Unit conversions and config management helpers
├── pyproject.toml          # Build configuration and dependency declarations
└── README.md
```
## License
MIT License

Copyright (c) 2026 Tomáš Taut

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
