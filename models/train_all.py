# train_multiple_models_structured.py

# Import necessary modules
from anomalib.data import Folder
# Import all models you plan to use
from anomalib.models.image.ganomaly import Ganomaly
from anomalib.models.image.efficient_ad import EfficientAd
# Add imports for other models here as needed, e.g.:
# from anomalib.models.image.<other_model> import <OtherModel>

from anomalib.engine import Engine
import os

# --- Function to train and test Ganomaly ---
def train_and_test_ganomaly(dataset: Folder):
    """Configures, trains, and tests the Ganomaly model."""
    print("\n--- Starting Ganomaly Pipeline ---")

    model = Ganomaly(
        train_batch_size=32,
    )

    engine = Engine(
        default_root_dir="Forschsem/results/", # Results will go into results/Ganomaly/...
        task=dataset.task,           # Match task from DataModule
        accelerator="gpu",
        devices=1,
        max_epochs=100,               # Set number of training epochs for Ganomaly
    )

    print(f"Starting Ganomaly training for {engine.max_epochs} epochs...")
    engine.train(
        model=model,
        datamodule=dataset # Use the shared DataModule
    )
    print("Ganomaly training complete.")

    # Step 4a: Evaluate Ganomaly Model
    print("Starting Ganomaly evaluation...")
    engine.test(datamodule=dataset) # Evaluate the trained Ganomaly model
    print("Ganomaly evaluation complete.")


# --- Function to train and test EfficientAd ---
def train_and_test_efficientad(dataset: Folder):
    """Configures, trains, and tests the EfficientAd model."""
    print("\n--- Starting EfficientAd Pipeline ---")

    # Step 2b: Initialize EfficientAd Model
    # Use the image size and task from the dataset DataModule
    model = EfficientAd(
        image_size=dataset.image_size,
        model_size="M", # Choose model size: "S", "M", or "L"
        train_batch_size=1,
        # Add other EfficientAd specific parameters here
    )

    # Step 3b: Train using Anomalib Engine for EfficientAd
    # Create ANOTHER NEW Engine instance for EfficientAd's training run
    engine = Engine(
        default_root_dir="results/", # Results will go into results/EfficientAd/...
        task=dataset.task,           # Match task from DataModule
        accelerator="gpu",
        devices=1,
        max_epochs=100,               # Set number of training epochs for EfficientAd
        versioned_dir=False
    )

    print(f"Starting EfficientAd training for {engine.max_epochs} epochs...")
    engine.train(
        model=model,
        datamodule=dataset # Use the shared DataModule
    )
    print("EfficientAd training complete.")

    # Step 4b: Evaluate EfficientAd Model
    print("Starting EfficientAd evaluation...")
    engine.test(datamodule=dataset) # Evaluate the trained EfficientAd model
    print("EfficientAd evaluation complete.")

# --- Add functions for other models in the same style ---
# def train_and_test_<other_model>(dataset: Folder):
#     """Configures, trains, and tests <other_model>."""
#     print(f"\n--- Starting <OtherModel> Pipeline ---")
#     # Initialize the other model
#     # Create a NEW Engine
#     # Call engine.train()
#     # Call engine.test()
#     print(f"<OtherModel> pipeline complete.")


# --- Main execution function ---
def main():
    """
    Sets up the data module and runs the training and testing pipeline
    for multiple anomaly detection models sequentially.
    """
    print("--- Setting up Data Module ---")
    # --- Step 1: Create DataModule (Shared) ---
    # Configure based on your powerful workstation
    dataset = Folder(
        root="D:/FS_Datensatz/Forschsem/data/strawberry_riseholme/",
        normal_dir="train",
        abnormal_dir="test/Unripe",
        normal_test_dir="test/Ripe",
        eval_batch_size=64,       # <-- Match train or higher
        num_workers=16,           # <-- High workers for multiprocessing (tune based on CPU cores
    )
    print("Data Module setup complete.")
    print("-" * 30) # Separator

    # --- Run Pipelines for Each Model Sequentially ---

    # Run Ganomaly pipeline
    train_and_test_ganomaly(dataset)
    print("-" * 30) # Separator

    # Run EfficientAd pipeline
    train_and_test_efficientad(dataset)
    print("-" * 30) # Separator

    # Add calls for other model pipelines here:
    # train_and_test_<other_model>(dataset)
    # print("-" * 30) # Separator

    print("\n--- All Model Pipelines Complete ---")


# --- Standard Python entry point ---
if __name__ == "__main__":
    # This block ensures the main() function is called only when the script
    # is executed directly, which is necessary when num_workers > 0 for
    # multiprocessing compatibility.
    main()