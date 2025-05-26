def main():
    from anomalib.data import Folder
    from anomalib.models.image.padim import Padim  # PaDiM-Modell importieren
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
        root="Forschsem/data/strawberry_riseholme_switched/",
        normal_dir="test/Unripe_normal",  # Normal Test Data
        abnormal_dir="test/Ripe_anormal",  # Anomalies
        normal_test_dir="train_unripe",  # Normal Train Data
        train_batch_size=8,  # Bilder die in den GPU-Speicher passen
        eval_batch_size=8,
        num_workers=8,  # kann ggf. auf 0 gesetzt werden zum Testen
    )

    # Modell definieren
    model = Padim(
        backbone="resnet18",  # Backbone für das Modell, -----------------------andere größere Möglichkeit: wide_resnet50_2
        pre_trained=False,     # Vortrainierte Gewichte verwenden
        layers=["layer1"],  # Zu verwendende Schichten ---------------------- hier kann auch noch layer3 getestet werden
    )

    # # Checkpoint-Callback konfigurieren
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\Padim\\checkpoints",
    #     filename="padim-{epoch:02d}-{val_loss:.6f}",
    #     monitor="val_loss",  # Überwache den Validierungsverlust
    #     mode="min",
    #     save_top_k=2,
    #     save_last=True,
    # )

    # # Frühes Stoppen
    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     patience=10,
    #     mode="min"
    # )

    # Schönere Fortschrittsanzeige
    rich_progress = RichProgressBar()

    # Logger definieren
    logger = TensorBoardLogger(
        save_dir="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\Padim\\logs",  # Speicherort der Logs
        name="padim"  # Name des Experiments
    )

    # Engine definieren
    engine = Engine(
        default_root_dir="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\Padim\\engine",
        accelerator="gpu",
        devices=1,
        #max_epochs=400, - PaDiM macht nur einen Forward-Pass
        callbacks=[rich_progress],  # Callback hinzufügen
        logger=logger  # Logger hinzufügen
    )

    # Training starten
    engine.fit(
        datamodule=datamodule,
        model=model
    )

    # Testen
    engine.test(
        model=model,
        datamodule=datamodule,
       # ckpt_path="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\Padim\\engine\\Padim\\Folder\\latest\\weights\\lightning\\model.ckpt"  # Optional: Pfad zu einem Checkpoint
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
