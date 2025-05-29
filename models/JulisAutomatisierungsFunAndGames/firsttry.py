import logging
from pathlib import Path
import torch
import pandas as pd

from anomalib.data import Folder
from anomalib.models import ReverseDistillation
from anomalib.engine import Engine
from anomalib.post_processing import PostProcessor
from anomalib.metrics import Evaluator
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger


def setup_logger(log_file: Path) -> logging.Logger:
    """
    Create and configure a logger writing to the specified file and to console.
    """
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def train_dataset(spec: dict, results_base: Path) -> dict:
    """
    Train, test, post-process and evaluate the model on a single dataset.
    If debug=True, runs a mini smoke test on CPU with reduced data and model size.
    """
    name = spec["name"]
    debug = spec.get("debug", False)

    # Reproducibility
    seed = spec.get("seed", 42)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Default or debug-specific parameters
    train_batch_size = spec.get("train_batch_size", 4 if debug else 32)
    eval_batch_size  = spec.get("eval_batch_size",  4 if debug else 32)
    num_workers      = spec.get("num_workers",      0 if debug else 8)
    max_epochs       = spec.get("max_epochs",       2 if debug else 200)
    backbone         = spec.get("backbone",        "resnet18")

    # Prepare directories
    dataset_dir = results_base / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(dataset_dir / "run.log")
    logger.info(f"Starting dataset: {name} (debug={debug})")

    try:
        # DataModule (keine Validierung)
        datamodule = Folder(
            name=name,
            root=spec["root"],
            normal_dir=spec["normal_dir"],
            abnormal_dir=spec["abnormal_dir"],
            normal_test_dir=spec["normal_test_dir"],
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )

        # Model
        model = ReverseDistillation(
            backbone=backbone,
            pre_trained=spec.get("pre_trained", False),
            layers=spec.get("layers", ["layer1", "layer2", "layer3"]),
        )

        # TensorBoard-Logger
        tb_logger = TensorBoardLogger(
            save_dir=str(dataset_dir / "logs"),
            name="rd_debug" if debug else "rd"
        )

        # Callbacks
        progress_bar = RichProgressBar()
        # EarlyStopping und Checkpointing auf train_loss_epoch
        early_stop = EarlyStopping(
            monitor="train_loss_epoch",
            patience=5,
            mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        checkpoint = ModelCheckpoint(
            dirpath=dataset_dir / "checkpoints",
            filename=f"{name}_epoch{{epoch:02d}}_trainLoss{{train_loss_epoch:.4f}}",
            monitor="train_loss_epoch",
            mode="min",
            save_top_k=1 if debug else 3,
            save_last=True,
        )

        # Trainer-Argumente (keine Validation)
        trainer_args = {
            "default_root_dir": str(dataset_dir / "engine"),
            "accelerator": "cpu" if debug else ("gpu" if torch.cuda.is_available() else "cpu"),
            "devices": 1,
            "max_epochs": max_epochs,
            "limit_train_batches": spec.get("limit_train_batches", 0.1 if debug else 1.0),
            "limit_val_batches": 0.0,   # keine Validierung
            "limit_test_batches": spec.get("limit_test_batches", 0.1 if debug else 1.0),
            "callbacks": [progress_bar, early_stop, lr_monitor, checkpoint],
            "logger": tb_logger,
        }

        engine = Engine(**trainer_args)

        # Training (nur Training-Schritte)
        engine.fit(datamodule=datamodule, model=model)

        # Test (unabhängig vom Debug-Flag)
        test_outputs = engine.test(model=model, datamodule=datamodule)

        # Postprocessing & Evaluation (nur bei Full-Run)
        metrics = {"dataset": name}
        if not debug:
            post_processor = PostProcessor()
            pp_results = post_processor(
                results=test_outputs,
                dataloader=datamodule.test_dataloader(),
            )
            evaluator = Evaluator(metrics=[
                "confusion_matrix", "auroc", "f1_score", "precision", "recall"
            ])
            eval_metrics = evaluator.evaluate(pp_results)
            metrics.update(eval_metrics)
            logger.info(f"Completed {name} with metrics: {metrics}")
            pd.DataFrame([metrics]).to_csv(dataset_dir / "metrics.csv", index=False)
        else:
            logger.info(f"Debug run for {name} completed.")

        return metrics

    except Exception as e:
        logger.exception(f"Error in dataset {name}")
        return {"dataset": name, "error": str(e)}


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

    from multiprocessing import freeze_support
    freeze_support()

    results_base = Path("results")
    results_base.mkdir(exist_ok=True)

    base_data = Path(__file__).parents[2] / "data" / "strawberry_riseholme"
    dataset_specs = [
        {
            "name": "strawberry_debug", #name kann gesetzt werden wie man lustig ist
         "root": str(base_data),
         "normal_dir": "train",
         "abnormal_dir": "test/Ripe",
         "normal_test_dir": "test/Unripe",
         "debug": True
         },
        
        {
            "name": "strawberry_full",
         "root": str(base_data),
         "normal_dir": "train",
         "abnormal_dir": "test/Ripe",
         "normal_test_dir": "test/Unripe",
         "debug": False}
        ,
    ]

    summary = [train_dataset(spec, results_base) for spec in dataset_specs]
    pd.DataFrame(summary).to_csv(results_base / "summary_results.csv", index=False)
    print(f"All runs completed. Summary saved to {results_base / 'summary_results.csv'}")
