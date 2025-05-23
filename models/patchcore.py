def main():
    from anomalib.data import Folder
    from anomalib.models.image.patchcore import Patchcore  # PatchCore-Modell importieren
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
        root="Forschsem/data/strawberry_riseholme_switched",
        normal_dir="train_unripe",  # Normal Train Data
        abnormal_dir="test/Ripe_anormal",  # Anomalies
        normal_test_dir="test/Unripe_normal",  # Normal Test Data
        train_batch_size=2,  # Bilder die in den GPU-Speicher passen
        eval_batch_size=2,
        num_workers=8,  # kann ggf. auf 0 gesetzt werden zum Testen
    )

    # Modell definieren
    model = Patchcore(
        backbone="resnet18",  # Leistungsstarkes Backbone-----------------------------------------wide_resnet50_2
        layers=["layer2", "layer3"],  # Tiefere Schichten für reichhaltige Merkmale ----- mal 1 und 2 ausporbieren.
        pre_trained=False,  # Vortrainierte Gewichte nicht verwenden
        coreset_sampling_ratio=0.01,  #------------------------------------------------Das höher Einstellen für höhere Genauigkeit?
        num_neighbors=32,  # Mehr Nachbarn für robustere KNN-Suche----------------------Das höher Einstellen für höhere Genauigkeit?
        post_processor=True  # Standard-PostProcessor aktivieren
    )
    
    # Configure pre-processor to reproduce paper settings - Werden die Bilder doof skaliert?
    # pre_processor = Patchcore.configure_pre_processor(
    # image_size=(128, 128), # kleinerer Resize
    

    # Threshold anpassen
    #model.post_processor.threshold = 0.4  # ------------------------------------------ Hier rumspielen: Threshold 

    # Schönere Fortschrittsanzeige
    rich_progress = RichProgressBar()

    # Logger definieren
    logger = TensorBoardLogger(
        save_dir="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\PatchCore\\logs",  # Speicherort der Logs
        name="patchcore"  # Name des Experiments
    )

    # Engine definieren
    engine = Engine(
        default_root_dir="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\PatchCore\\engine",
        accelerator="gpu",
        devices=1,
        #max_epochs=400, - Patchcore macht nur einen Forward-Pass
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
        #ckpt_path="C:\\Users\\Matthis\\Documents\\GitHub\\Anoamlib\\results\\PatchCore\\engine\\Patchcore\\Folder\\latest\\weights\\lightning\\model.ckpt"  # Optional: Pfad zu einem Checkpoint
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
