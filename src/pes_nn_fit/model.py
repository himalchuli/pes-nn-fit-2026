import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import Dataset
from .preprocess import Preprocessor
import logging
import copy
import os

logger = logging.getLogger(__name__)


class Model(nn.Sequential):
    """
    Model class which inherits properties from `torch.nn.Sequential`.

    Attributes:
        input_size (str): Size of the input layer.
        hyperparams (dict): Hyperparameters of the model.
            Keyword attributes:
            - layers (str): In form "x,y,z,..".
            - activation (str): Activation function between linear layers.
            Options: `'tanh'`.
    """

    def __init__(self, input_size: str, hyperparams: dict) -> None:
        """
        Construct a feed-forward neural network implemented as ``torch.nn.Sequential``.
        All linear layers use Xavier uniform initialization.

        Args:
            input_size (int): Number of input features (size of the first layer).
            hyperparams (dict): Model hyperparameters, typically from
                ``config['hyperparams']['model']``.

                Expected keys:
                - layers (str): Sizes of hidden layers. Comma-separated string
                (e.g., ``"50,10"``).
                - activation (str): Activation function name (currently ``"tanh"``).

        """
        self.input_size = input_size
        self.hyperparams = hyperparams

        # layer input can be both [50,10] and "50,10"
        layers_raw = hyperparams["layers"]
        # transfrom from string to list if needed
        layer_sizes = (
            [int(x) for x in layers_raw.split(",")]
            if isinstance(layers_raw, str)
            else list(layers_raw)
        )
        if hyperparams["activation"] == "tanh":
            activation = nn.Tanh
        else:
            raise ValueError(
                f"Unsupported activation '{hyperparams['activation']}'. Allowed: ['tanh']"
            )

        layers = []
        in_size = input_size

        for layer_size in layer_sizes:
            layer = nn.Linear(in_size, layer_size)
            nn.init.xavier_uniform_(layer.weight)
            layers.extend([layer, activation()])
            in_size = layer_size

        # append the last layer
        layer = nn.Linear(in_size, 1)
        nn.init.xavier_uniform_(layer.weight)
        layers.append(layer)

        # create the model
        super().__init__(
            *layers
        )  # using sequential, forward() is implemented by itself

        logger.info(
            "Model initialized. Layers: %s, Activation: %s",
            layer_sizes,
            hyperparams["activation"],
        )


class TrainingResult:
    "Helper class for results of training."

    def __init__(self, stage, train_losses, val_losses):
        self.stage = stage  # 'adam' or 'lm'
        self.train_losses = train_losses
        self.val_losses = val_losses

        self.final_train = train_losses[-1]
        self.final_val = val_losses[-1] if val_losses else None


class Trainer:
    """
    Train a model and record training statistics.

    The trainer manages optimization, evaluation, and logging of losses
    across training, validation, and test datasets according to provided
    hyperparameters.
    """

    def __init__(
        self, hyperparams: dict, device: str = "cpu", dtype: str = None
    ) -> None:
        """
        Initialize the trainer.

        Args:
            device (str or torch.device, optional): Device used for training
                and inference. Defaults to ``"cpu"``.
            dtype (str): dtype used during training. Must be supported by PyTorch.
            hyperparams (dict): Training configuration, typically from
                ``config['hyperparams']['trainer']``.
        """

        self.hyperparams = hyperparams
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float64

        logger.info("Trainer initialized.")

    def _to_tensor(self, array) -> torch.Tensor:
        "Array to tensor with class device and dtype."
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def train(
        self,
        model,
        optimizer_type,
        trainset,
        valset=None,
        trial=None,
        checkpoint_path=None,
        checkpoint_interval=10,
    ) -> TrainingResult:
        """
        Train the model using Levenberq-Marquardt algorithm
        Args:
            model : torch.nn.Module
                Model to be optimized.
            optimizer_type (str): which optimizer to use, `"lm"` or `"adam"`
            trainset (`Dataset obj`): data for training
            valset (`Dataset obj`, optional): data for validation
            testset (`ndarray`, optional): data for test
            trial : trial obj from optuna
            checkpoint_path (str):
                Path to checkpoints during saving and loading, if `None`
                checkpointing is disabled.
            checkpoint_interval (int): number of epochs after checkpoint is saved
        Returns:
            TrainingResult: holds data about training
        """
        import optuna

        start_epoch = 0
        # arrays for losses
        train_losses = []
        val_losses = []
        # early stopping initialization
        early_cfg = self.hyperparams["early_stopping"]
        early_best_val = float("inf")
        early_best_state = None
        early_patience_counter = 0

        # Transform data and targets to tensors
        X = self._to_tensor(trainset.data)
        y = self._to_tensor(trainset.target)

        if valset is not None:
            X_val = self._to_tensor(valset.data)
            y_val = self._to_tensor(valset.target)
            # criterion for validation
            val_criterion = nn.MSELoss()

        if optimizer_type == "lm":
            import torch_levenberg_marquardt as tlm

            criterion = tlm.loss.MSELoss()
            module_lm = tlm.training.LevenbergMarquardtModule(
                model=model,
                loss_fn=criterion,
                learning_rate=self.hyperparams["lm"]["lr"],
                attempts_per_step=self.hyperparams["lm"]["attempts_per_step"],
                use_vmap=self.hyperparams["lm"]["use_vmap"],
                max_batch_size=self.hyperparams["lm"]["batch_size"],
                solve_method=self.hyperparams["lm"]["solve_method"],
            )

            logger.info("Training by Levenberg-Marquardt algorithm...")

        elif optimizer_type == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=self.hyperparams["adam"]["lr"]
            )
            criterion = nn.MSELoss()
            trainloader_adam = torch.utils.data.DataLoader(
                trainset,
                batch_size=self.hyperparams["adam"]["batch_size"],
                shuffle=True,
            )

            logger.info("Training by Adam optimizer...")

        # load from checkpoints if enabled and was saved
        if checkpoint_path and checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if optimizer_type == "adam":
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if optimizer_type == "lm":
                # save state dict for module_lm
                module_lm.flat_params.copy_(ckpt["flat_params"])
                module_lm.damping_strategy.damping_factor = ckpt["damping_factor"]
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            train_losses = ckpt["train_losses"]
            val_losses = ckpt["val_losses"]
            early_best_val = ckpt["early_best_val"]
            early_patience_counter = ckpt["early_patience_counter"]
            logger.info("Resumed from epoch %d", start_epoch)

        for epoch in range(start_epoch, self.hyperparams[optimizer_type]["max_epoch"]):
            model.train()
            if optimizer_type == "lm":
                _, train_loss, _, _ = module_lm.training_step(X, y)
                train_losses.append(train_loss.item())

            elif optimizer_type == "adam":
                running_loss = 0.0
                for inputs, labels in trainloader_adam:
                    inputs = self._to_tensor(inputs)
                    labels = self._to_tensor(labels)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    train_loss = criterion(outputs, labels)
                    train_loss.backward()
                    optimizer.step()
                    running_loss += train_loss.item()
                # epoch_loss should be average loss for all batches
                epoch_loss = running_loss / len(trainloader_adam)
                train_losses.append(epoch_loss)
                running_loss = 0.0

            if valset is not None:
                model.eval()
                with torch.no_grad():
                    preds_val = model(X_val)
                    val_loss = val_criterion(preds_val, y_val).item()
                    val_losses.append(val_loss)
                    # Optuna pruning (trial-level)
                    if trial is not None and optimizer_type == "lm":
                        trial.report(val_loss, step=epoch)
                        if trial.should_prune():
                            logger.info("Trial %d pruned.", trial.number)
                            raise optuna.TrialPruned()

                    # check early stopping
                    if early_cfg["enabled"]:
                        if val_loss < early_best_val:
                            early_best_val = val_loss
                            early_best_state = copy.deepcopy(model.state_dict())
                            early_patience_counter = 0
                        else:
                            early_patience_counter += 1

                        if early_patience_counter >= early_cfg["patience"]:
                            logger.info(
                                "Early stopping triggered at epoch %d | Best Val Loss: %.3e",
                                epoch + 1,
                                early_best_val,
                            )
                            break
            else:
                # validaton set not available -> nan
                val_loss = float("nan")

            # print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d | Train Loss: %.3e | Val Loss: %.3e",
                    epoch + 1,
                    self.hyperparams[optimizer_type]["max_epoch"],
                    train_loss,
                    val_loss,
                )

            # save progress every checkpoint_interval epochs
            if checkpoint_path and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                if optimizer_type == "adam":
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "early_best_val": early_best_val,
                            "early_patience_counter": early_patience_counter,
                        },
                        checkpoint_path,
                    )

                elif optimizer_type == "lm":
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "flat_params": module_lm.flat_params.clone(),  # replaces optimizer_state_dict
                            "damping_factor": module_lm.damping_strategy.damping_factor.clone(),
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "early_best_val": early_best_val,
                            "early_patience_counter": early_patience_counter,
                        },
                        checkpoint_path,
                    )

                logger.debug(
                    "Saved training progress after epoch %d to %s.",
                    epoch,
                    checkpoint_path,
                )

        if early_cfg["enabled"] and early_best_state is not None:
            model.load_state_dict(early_best_state)
            logger.info("Best model weights from early stopping restored.")

        logger.info("Training completed.")
        return TrainingResult(optimizer_type, train_losses, val_losses)


class Pipeline:
    """
    Handles model initialization, preprocessing, training and prediction.

    Attributes:
        device (str): Selected computation device.
        hyperparams (dict): Stored hyperparameters.
        adam_losses (Optional[list]): Tracks losses from Adam optimization phase.
        lm_losses (Optional[list]): Tracks losses from LM optimization phase.
        model (Optional[Any]): Model instance initialized later in the pipeline.
        seed (int): Random seed for reproducibility.
        dtype (torch.dtype): Torch dtype corresponding to the selected precision.
    """

    def __init__(
        self,
        hyperparams: dict,
        device: str = "cpu",
        dtype: str = "float64",
        seed: int = 42,
    ) -> None:
        """
        Create a pipeline.

        Args:
            hyperparams (dict): Hyperparameters from the configuration file. Expected
                to include all settings required to initialize and train the model
                (e.g., learning rates, architecture parameters, optimizer settings).
            device (str, optional): Target device for computation. Typically "cpu"
                or "cuda". Defaults to "cpu".
            dtype (str, optional): Numerical precision to use for tensors. Supported
                values are "float32" and "float64". Defaults to "float64".
            seed (int, optional): Random seed for reproducibility across runs.
                Defaults to 42.
        """
        self.device = device
        self.hyperparams = hyperparams
        self.adam_losses = None
        self.lm_losses = None
        self.model = None
        self.seed = seed

        # dtype
        if dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsuported dtype {dtype}")

        logger.info(
            "Pipeline initialized. Device: %s | dtype: %s",
            self.device,
            dtype,
        )

    def fit(self, trainset, valset=None, trial=None, checkpoint_dir=None):
        """
        Fit the pipeline by preprocessing data and training the model.

        Args:
            trainset: Training dataset with `.data` and `.target` attributes.
            valset (optional): Validation dataset used for evaluation during training.
            trial (optional): Optimization trial object (e.g., for hyperparameter tuning).
            checkpoint_dir (optional): Directory to save or load training checkpoints.

        Returns:
            self: Fitted pipeline instance.
        """
        pre_cfg = self.hyperparams["preprocess"]
        model_cfg = self.hyperparams["model"]
        trainer_cfg = self.hyperparams["trainer"]

        # Preprocessing
        self.pre = Preprocessor(pre_cfg)
        self.pre.fit(trainset.data, trainset.target)
        Xtr, ytr = self.pre.transform(trainset.data, trainset.target)
        trainset_trans = Dataset(array_tuple=(Xtr, ytr), seed=self.seed)

        if valset is not None:
            Xval, yval = self.pre.transform(valset.data, valset.target)
            valset_trans = Dataset(array_tuple=(Xval, yval), seed=self.seed)
        else:
            valset_trans = None

        # Train the model
        input_size = trainset_trans.data.shape[1]  # number of features (columns)
        self.model = Model(input_size, model_cfg)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        trainer = Trainer(
            trainer_cfg,
            device=self.device,
            dtype=self.dtype,
        )

        # initialize the checkpoint directory if given
        if checkpoint_dir:
            adam_ckpt = checkpoint_dir / "adam.pt"
            lm_ckpt = checkpoint_dir / "lm.pt"
        else:
            adam_ckpt = None
            lm_ckpt = None

        if trainer_cfg["adam"]["train"]:
            self.adam_result = trainer.train(
                self.model,
                "adam",
                trainset_trans,
                valset_trans,
                trial,
                checkpoint_path=adam_ckpt,
            )
        if trainer_cfg["lm"]["train"]:
            self.lm_result = trainer.train(
                self.model,
                "lm",
                trainset_trans,
                valset_trans,
                trial,
                checkpoint_path=lm_ckpt,
            )
        return self

    def predict(self, X_raw):
        """
        Generate predictions from raw input data.

        Args:
            X_raw: Raw input features.

        Returns:
            np.ndarray: Predictions in the original target space.
        """
        if self.model is None:
            raise RuntimeError("Pipeline must be fitted before prediction")
        X_trans = self.pre.transform(X_raw)
        X_trans = torch.as_tensor(X_trans, dtype=self.dtype, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y_trans = self.model(X_trans)
        y_trans = y_trans.detach().cpu().numpy()
        y_raw = self.pre.inverse_transform_y(y_trans)
        return y_raw

    def __call__(self, X_raw):
        """Alias for `Pipeline.predict`"""
        return self.predict(X_raw)

    def save(self, path):
        """
        Save the fitted pipeline.

        Args:
            path (str or Path): File path to save the checkpoint.
        """
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted pipeline.")

        if self.dtype == torch.float64:
            dtype = "float64"
        elif self.dtype == torch.float32:
            dtype = "float32"

        checkpoint = {
            "hyperparams": self.hyperparams,
            "model_state_dict": self.model.state_dict(),
            "model_input_size": self.model.input_size,
            "preprocessor_state": self.pre.__dict__,  # assumes pure-data attributes
            "adam_result": (
                self.adam_result.__dict__ if hasattr(self, "adam_result") else None
            ),
            "lm_result": (
                self.lm_result.__dict__ if hasattr(self, "lm_result") else None
            ),
            "dtype": dtype,
            "seed": self.seed,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

        logger.info("Pipeline saved at %s.", path)

    @classmethod
    def load(cls, path, device="cpu", dtype=None):
        """
        Load a pipeline.

        Args:
            path (str or Path): Path to the saved checkpoint.
            device (str, optional): Target device. Defaults to "cpu".
            dtype (str, optional): Override stored dtype.

        Returns:
            Pipeline: Reconstructed pipeline instance.
        """
        checkpoint = torch.load(path, map_location=device)

        if dtype is None:
            dtype = checkpoint["dtype"]

        pipe = cls(checkpoint["hyperparams"], device=device, dtype=dtype)

        # rebuild model
        input_size = checkpoint["model_input_size"]
        pipe.model = Model(input_size, pipe.hyperparams["model"])
        pipe.model.load_state_dict(checkpoint["model_state_dict"])

        torch_dtype = pipe.dtype
        pipe.model = pipe.model.to(device=device, dtype=torch_dtype)
        pipe.device = device

        # rebuild preprocessor
        pipe.pre = Preprocessor(pipe.hyperparams["preprocess"])
        pipe.pre.__dict__.update(checkpoint["preprocessor_state"])

        # restore training results if available
        if checkpoint["adam_result"]:
            pipe.adam_result = TrainingResult(
                stage="adam",
                train_losses=checkpoint["adam_result"]["train_losses"],
                val_losses=checkpoint["adam_result"]["val_losses"],
            )
        if checkpoint["lm_result"]:
            pipe.lm_result = TrainingResult(
                stage="lm",
                train_losses=checkpoint["lm_result"]["train_losses"],
                val_losses=checkpoint["lm_result"]["val_losses"],
            )

        logger.info("Pipeline loaded from %s to %s.", path, device)

        return pipe
