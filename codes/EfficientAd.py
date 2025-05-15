# train_script.py
# Import the Folder DataModule (not the base dataset)
from anomalib.data import Folder
from anomalib.models.image.ganomaly import Ganomaly
from anomalib.models.image.efficient_ad import EfficientAd
from anomalib.engine import Engine

# Step 1: Create DataModule
# Use the Folder DataModule for handling custom directory structure
# Pass the task here. Also, CORRECT YOUR ROOT PATH for Windows
dataset = Folder(
    name="Folder",
    root="C:/Users/Lenovo/Documents/02_repos/Forschungsseminar/Anomalib/data", # <-- CORRECT THIS PATH to your actual dataset root
    normal_dir="train/good",  # <-- Specify the relative path to training good images
    abnormal_dir="test/bad", # <-- Specify the relative path to test abnormal images (or test/your_anomaly_type)
    normal_test_dir="test/good", # <-- Specify the relative path to test good images
    #task="classification",    # <-- Specify the task here (classification, segmentation)
    #image_size=96,            # Resize to a consistent shape
    # transformations are often handled automatically by the DataModule based on task and image_size
    # If you need custom transforms, you'd define them and pass them here.
    # Leaving transform_config_train/eval=None might default to standard transforms or no transforms depending on version.
    # Let's remove them for clarity unless you have specific needs.
    # transform_config_train=None,
    # transform_config_eval=None,
    train_batch_size=1, # <-- Start with a small batch size for your GPU
    eval_batch_size=4,  # <-- Start with a small batch size for your GPU
    num_workers=0 # Keep low for debugging / limited resources
)

# Step 2: Initialize Model
# Ganomaly model also needs the image size and number of channels
# model = Ganomaly(
#     latent_vec_size=100, # Size of the latent vector
#     batch_size=2 # Has to match with the DataModule batch size
# )

model = EfficientAd()

# Step 3: Train using Anomalib Engine
# Pass the task to the Engine as well
engine = Engine(
    default_root_dir="results/",
    #task="classification", # <-- Specify the task for the Engine
    accelerator="gpu",     # <-- Explicitly set to use GPU
    devices=1,             # <-- Use 1 GPU
    max_epochs=10,          # <-- Reduce epochs for a quick test run
    #versioned_dir = False
)

# Train the model using the DataModule
engine.train(
    model=model,
    datamodule=dataset
)

# Step 4: Evaluate Model
# Use engine.test(datamodule=dataset) after training.
# The engine holds the trained model internally.
# Engine.test is a class method for testing *any* model with *any* datamodule,
# but for testing the model *just trained* by this engine instance, use engine.test().
engine.test(datamodule=dataset)

print("Training and testing complete.")