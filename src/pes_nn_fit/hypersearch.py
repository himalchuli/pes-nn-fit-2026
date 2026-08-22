from .model import Pipeline
import optuna
from src.utils.helpers import sample_config
import logging
import shutil

logger = logging.getLogger(__name__)


class HyperOptimizer:
    """
    Perform a search for best hyperparameters and keep information about
    progress. Uses Optuna.
    """

    def __init__(self, trainset, valset, config: dict, device: str = "cpu") -> None:
        """
        Initialize the hyperparameter optimizer.

        Args:
            trainset (Dataset obj): Training dataset.
            valset (Datset obj): Validation dataset.
            config (dict): Main config dictionary for HyperOptimizer.
                Holds both internal config for the optimizer (number of trials,
                pruner,...) and also it defines the hyperparameter space to be searched.
            device (str or torch.device, optional): Device used for training.
                Default: `'cpu'`.
        """
        self.device = device
        self.trainset = trainset
        self.valset = valset
        self.seed = config["runtime"]["seed"]
        self.dtype = config["runtime"]["dtype"]
        self.hyperparams = config["hyperparams"]
        self.cfg_optuna = config["optuna"]  # config for optuna
        self.checkpoint_dir = None  # is later filled in self.optimize()

    def _create_pruner(self) -> None:
        """Create a pruner as is specified in `HyperOptimizer.config`."""

        prune_cfg = self.cfg_optuna["prune"]
        name = prune_cfg.get("pruner", "median")

        if name is None:
            return None
        elif name == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=prune_cfg["n_startup_trials"],
                n_warmup_steps=prune_cfg["n_warmup_steps"],
                interval_steps=prune_cfg["interval_steps"],
            )
        else:
            raise ValueError(f"Unknown pruner: {name}")

    def objective(self, trial) -> float:
        """Optuna objective to minimize. Optuna trial as input."""
        # get this trial's params
        params = sample_config(trial, self.hyperparams)

        logger.info("Starting trial %d", trial.number)
        logger.info("Trial %d parameters: %s", trial.number, params)
        # Per-trial checkpoint directory
        trial_ckpt_dir = (
            self.checkpoint_dir / f"trial_{trial.number}"
            if self.checkpoint_dir is not None
            else None
        )

        # create a pipeline with correct hyperparams, fit it and obtain validation loss
        pipe = Pipeline(params, device=self.device, dtype=self.dtype, seed=self.seed)
        pipe.fit(
            self.trainset,
            self.valset,
            trial=trial,
            checkpoint_dir=trial_ckpt_dir,
        )
        val_loss = pipe.lm_result.final_val

        # Clean up checkpoints for this trial now that it completed successfully
        if trial_ckpt_dir is not None and trial_ckpt_dir.exists():
            shutil.rmtree(trial_ckpt_dir)
            logger.info("Trial %d checkpoints removed.", trial.number)

        logger.info(
            "Trial %d completed | Validation Loss: %.6e", trial.number, val_loss
        )

        return val_loss

    def optimize(
        self,
        storage_path: str = None,
        load_if_exists: bool = False,
        checkpoint_dir: str = None,
    ) -> optuna.study.Study:
        """
        Main function for hyperparameter optimization. Finds the best
        hyperparameters of the model.

        Args:
            storage_path (str, optional): Full path to saved optuna study.
                If `None`, then the study is not saved after the run is ended.
                Default: `None`.
            load_if_exists (bool, optional): Load existing study if it exists.
            checkpoint_dir (str, optional):
                Checkpoint directory used in `Pipeline.fit`. Default: `None`.

        """

        logger.info(
            "Starting hyperparameter optimization | Trials: %d",
            self.cfg_optuna["n_trials"],
        )

        # save checkpoint_dir as attribute to be used in self.objective()
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = self._create_pruner()
        study_name = self.cfg_optuna["study_name"]

        if storage_path is None:
            storage_name = (
                None  # if storage_path is None, the study will not be saved longterm
            )
        else:
            storage_path.mkdir(parents=True, exist_ok=True)
            # usually should exp_root/models be used in scripts as storage_path
            storage_name = f"sqlite:///{storage_path}/{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
            load_if_exists=load_if_exists,
        )

        # If resumed complete only the the remaining part of the trials
        n_trials_total = self.cfg_optuna["n_trials"]
        n_completed = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        n_pruned = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        )
        n_remaining = n_trials_total - (n_completed + n_pruned)

        logger.info("Finished trials: %d / %d", n_completed + n_pruned, n_trials_total)

        if n_remaining <= 0:
            logger.info("All trials already completed, skipping optimization.")
        else:
            study.optimize(self.objective, n_trials=n_remaining)

        self.study = study

        logger.info("Hyperparameter optimization completed.")
        logger.info(
            "Best trial: %d | Best validation loss: %.6e",
            study.best_trial.number,
            study.best_value,
        )
        logger.info("Best parameters: %s", study.best_trial.params)

        return study
