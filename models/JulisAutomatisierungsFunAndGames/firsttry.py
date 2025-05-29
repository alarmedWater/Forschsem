import logging
from pathlib import Path
import torch
import pandas as pd

from anomalib.data import Folder
from anomalib.models import ReverseDistillation
from anomalib.engine import Engine
from anomalib.post_processing import PostProcessor
from anomalib.metrics import Evaluator
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def setup_logger(log_file: Path) -> logging.Logger:
    """
    Create and configure a logger writing to the specified file.
    """
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def train_dataset(spec: dict, results_base: Path) -> dict:
    """
    Train, test, post-process and evaluate the model on a single dataset.
    If debug=True, runs a fast smoke test on CPU.
    Returns a dict of metrics or error info.
    """
    name = spec["name"]
    debug = spec.get("debug", False)

    # Prepare directories
    dataset_dir = results_base / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log_file = dataset_dir / "run.log"
    logger = setup_logger(log_file)
    logger.info(f"Starting dataset: {name} (debug={debug})")

    try:
        # DataModule
        datamodule = Folder(
            name=name,
            root=spec["root"],
            normal_dir=spec["normal_dir"],
            abnormal_dir=spec["abnormal_dir"],
            normal_test_dir=spec["normal_test_dir"],
            train_batch_size=spec.get("train_batch_size", 32),
            eval_batch_size=spec.get("eval_batch_size", 32),
            num_workers=spec.get("num_workers", 8),
        )

        # Model
        model = ReverseDistillation(
            backbone=spec.get("backbone", "resnet18"),
            pre_trained=spec.get("pre_trained", False),
            layers=spec.get("layers", ["layer1", "layer2", "layer3"]),
        )

        # Logger
        tb_logger = TensorBoardLogger(
            save_dir=str(dataset_dir / "logs"),
            name="rd_debug" if debug else "rd"
        )

        # Callbacks
        progress_bar = RichProgressBar()
        checkpoint_callback = ModelCheckpoint(
            dirpath=dataset_dir / "checkpoints",
            filename=f"{name}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )

        # Trainer arguments
        trainer_args = {
            "default_root_dir": str(dataset_dir / "engine"),
            "accelerator": "cpu" if debug else "gpu",
            "devices": 1,
            "max_epochs": 1 if debug else spec.get("max_epochs", 200),
            "callbacks": [progress_bar, checkpoint_callback],
            "logger": tb_logger,
        }

        if debug:
            # fast_dev_run: train+val+test with a single batch
            trainer_args.update({
                "fast_dev_run": True,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "limit_test_batches": 1,
            })

        # Create Engine (Trainer)
        engine = Engine(**trainer_args)

        # Training
        engine.fit(datamodule=datamodule, model=model)

        # Testing (skip if fast_dev_run)
        test_outputs = []
        if not debug:
            test_outputs = engine.test(model=model, datamodule=datamodule)

        # Evaluation (full run only)
        metrics = {"dataset": name}
        if not debug:
            # Post-process
            post_processor = PostProcessor()
            pp_results = post_processor(
                results=test_outputs,
                dataloader=datamodule.test_dataloader(),
            )

            # Compute metrics
            evaluator = Evaluator(metrics=[
                "confusion_matrix",
                "auroc",
                "f1_score",
                "precision",
                "recall",
            ])
            eval_metrics = evaluator.evaluate(pp_results)
            metrics.update(eval_metrics)
            logger.info(f"Completed {name} with metrics: {metrics}")

            # Save detailed metrics
            pd.DataFrame([metrics]).to_csv(
                dataset_dir / "metrics.csv", index=False
            )
        else:
            logger.info(f"Debug run for {name} completed.")

        return metrics

    except Exception as e:
        logger.exception(f"Error in dataset {name}")
        return {"dataset": name, "error": str(e)}


if __name__ == "__main__":
    # CUDA optimizations
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

    from multiprocessing import freeze_support
    freeze_support()

    # Base results directory
    results_base = Path("results")
    results_base.mkdir(exist_ok=True)

    # Dataset specifications, jetzt mit rein ASCII-Pfaden

    base_data = Path(__file__).parents[2] / "data" / "strawberry_riseholme"
    dataset_specs = [
        {
            "name": "strawberry_riseholme0",
            "root": str(base_data),            # …/data/strawberry_riseholme
            "normal_dir": "train",             # Trainingsdaten
            "abnormal_dir": "test/Ripe",       # defekte Früchte
            "normal_test_dir": "test/Unripe",  # normale Früchte zum Testen
            "debug": True,
        },
                {
            "name": "strawberry_riseholme1",
            "root": str(base_data),            # …/data/strawberry_riseholme
            "normal_dir": "train",             # Trainingsdaten
            "abnormal_dir": "test/Ripe",       # defekte Früchte
            "normal_test_dir": "test/Unripe",  # normale Früchte zum Testen
            "debug": True,
        },
                        {
            "name": "strawberry_riseholme2",
            "root": str(base_data),            # …/data/strawberry_riseholme
            "normal_dir": "train",             # Trainingsdaten
            "abnormal_dir": "test/Ripe",       # defekte Früchte
            "normal_test_dir": "test/Unripe",  # normale Früchte zum Testen
            "debug": True,
        },
    
         
    ]

    # Execute training for each dataset
    summary = [train_dataset(spec, results_base) for spec in dataset_specs]

    # Save summary
    pd.DataFrame(summary).to_csv(
        results_base / "summary_results.csv", index=False
    )
    print(
        f"All runs completed. Summary saved to {results_base / 'summary_results.csv'}"
    )
