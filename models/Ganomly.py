def main():
    from anomalib.data import Folder
    from anomalib.models.image.ganomaly import Ganomaly
    from anomalib.engine import Engine

    dataset = Folder(
        name="Folder",
        root="D:/FS_Datensatz/Forschsem/data/strawberry_riseholme/",
        normal_dir="train",
        abnormal_dir="test/Unripe",
        normal_test_dir="test/Ripe",
        train_batch_size=32,
        eval_batch_size=64,
        num_workers=8  # kannst du ggf. auf 0 setzen zum Testen
    )

    model = Ganomaly()

    engine = Engine(
        default_root_dir="results/",
        accelerator="gpu",
        devices=1,
        max_epochs=10,
    )

    engine.train(
        model=model,
        datamodule=dataset
    )

    engine.test(datamodule=dataset)

    print("Training and testing complete.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
