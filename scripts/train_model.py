import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

# my modules
from pes_nn_fit.dataset import Dataset
from pes_nn_fit.hypersearch import HyperOptimizer
from pes_nn_fit.model import Pipeline
from pes_nn_fit.utils.helpers import merge_config, unflatten_params
from pes_nn_fit.utils.setup import setup_experiment

# parser to parse --config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)


def main(args: argparse.Namespace):

    config, logger, device, exp_root = setup_experiment(Path(args.config).resolve())
    cfg_data = config["data"]

    # load dataset
    dataset = Dataset(
        file_location=cfg_data["data_file"], file_type=cfg_data["data_type"]
    )
    molecule = config["hyperparams"]["preprocess"]["molecule"]
    dataset.zmatrix2interatom(molecule=molecule)

    train_size, val_size, test_size = config["dataset"]["split_ratios"]
    trainset, valset, testset = dataset.split(
        split_type=config["dataset"]["split_type"],
        val_size=val_size,
        test_size=test_size,
    )
    trainset.save(exp_root / cfg_data["trainset_file"])  # save test set for later use
    valset.save(exp_root / cfg_data["valset_file"])  # save test set for later use
    testset.save(exp_root / cfg_data["testset_file"])  # save test set for later use

    t0 = time.perf_counter()

    optimizer = HyperOptimizer(trainset, valset, config, device)
    study = optimizer.optimize()
    best_params_flat = study.best_params
    best_params_nested = unflatten_params(best_params_flat)
    best_config = merge_config(config["hyperparams"], best_params_nested)

    trainset_full = trainset.combine(valset)
    pipeline = Pipeline(best_config, device=device, dtype=config["runtime"]["dtype"])
    pipeline.fit(trainset_full, testset)

    t1 = time.perf_counter()

    # save the pipeline
    pipeline.save(exp_root / cfg_data["model_file"])

    predictions = pipeline.predict(testset.data)

    # Comparison of my results with PES-Learn
    def calc_errors(real, predicted, energies_min=0):
        # Prepare a plot of energy vs prediction error
        relative = real - energies_min
        errors = predicted - real
        return relative, errors

    relative, errors = calc_errors(testset.target, predictions, np.min(predictions))
    rmse = np.sqrt(mean_squared_error(testset.target, predictions))

    logger.info("RMSE of of test energies is: {:.2e} Hartree\n".format(rmse))
    logger.info(
        "\nBest Hyperparameters: {} \n \
        Best MSE: {:.2e}".format(study.best_params, study.best_trial.value)
    )
    logger.info(f"Elapsed time: {t1 - t0} s")


if __name__ == "__main__":
    main_args = parser.parse_args()
    main(main_args)
