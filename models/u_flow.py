def main():
    from anomalib.data import Folder
    from anomalib.models.image import Uflow  # U-Flow-Modell importieren
    from anomalib.engine import Engine
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
    from lightning.pytorch.loggers import TensorBoardLogger
    import torch

    # CUDA-Optimierungen
    torch.set_float32_matmul_precision('medium')  # Enable Tensor Cores
    torch.backends.cudnn.benchmark = True       # Optimize for fixed input sizes

    # Datensatz definieren
    datamodule = Folder(
        name="Folder",
        root="Forschsem/data/strawberry_riseholme/",
        normal_dir="train",  # Normal Train Data
        abnormal_dir="test/Unripe",  # Anomalies
        normal_test_dir="test/Ripe",  # Normal Test Data
        train_batch_size=64,  # Bilder die in den GPU-Speicher passen
        eval_batch_size=64,
        num_workers=8,  # kann ggf. auf 0 gesetzt werden zum Testen
    )

    # Modell definieren
    model = Uflow( # Hat unterschiedliche Backbones , können Pretrained sein oder nicht
    )

    # # Checkpoint-Callback konfigurieren
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\UFlow\\checkpoints",
    #     filename="uflow-{epoch:02d}-{val_loss:.6f}",
    #     monitor="val_loss",  # Überwache den Validierungsverlust
    #     mode="min",
    #     save_top_k=2,
    #     save_last=True,
    # )

    # Frühes Stoppen
    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     patience=10,
    #     mode="min"
    # )

    # Schönere Fortschrittsanzeige
    rich_progress = RichProgressBar()

    # Logger definieren
    logger = TensorBoardLogger(
        save_dir="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\UFlow\\logs",  # Speicherort der Logs
        name="uflow"  # Name des Experiments
    )

    # Engine definieren
    engine = Engine(
        default_root_dir="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\UFlow\\engine",
        accelerator="gpu",
        devices=1,
        max_epochs=400,
        callbacks=[rich_progress],  # Callback hinzufügen
        logger=logger  # Logger hinzufügen
    )

    # Training starten
    # engine.fit(
    #     datamodule=datamodule,
    #     model=model
    # )

    # Testen
    engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\UFlow\\engine\\Uflow\\Folder\\v2\\weights\\lightning\\model.ckpt"  # Optional: Pfad zu einem Checkpoint
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
