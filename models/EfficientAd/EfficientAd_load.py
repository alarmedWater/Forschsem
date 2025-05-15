from anomalib.models.image.efficient_ad import EfficientAd
# You'll also need classes for data handling and inference if you plan to run predictions
from anomalib.data import Folder
from anomalib.engine import Engine # Engine has prediction methods
import torch

# --- Define the path to the saved checkpoint ---
# You'll need to find the exact path after training
EFFICIENTAD_CHECKPOINT_PATH = "results/EfficientAd/Folder/latest/model.ckpt" # Example path (adjust as needed)
# Or if using versioned_dir=True (default behavior):
# GANOMALY_CHECKPOINT_PATH = "results/Ganomaly/Folder/v0/weights/last.ckpt" # Example path (adjust as needed)


# --- Load the model from the checkpoint ---
try:
    # You call load_from_checkpoint on the Model Class itself
    loaded_efficientad_model = EfficientAd.load_from_checkpoint(
        EFFICIENTAD_CHECKPOINT_PATH,
        # You might need to pass model-specific arguments here IF they are
        # required by the model's __init__ but NOT saved in the checkpoint.
        # However, Anomalib models usually save these. Let's try without first.
        # For Ganomaly, potentially: latent_vec_size=100, n_features=... etc.
        # Batch size is typically handled by the DataLoader/Engine during inference, not a model init arg when loading.
    )
    print(f"Successfully loaded Ganomaly model from {EFFICIENTAD_CHECKPOINT_PATH}")

    # Put the model in evaluation mode
    loaded_efficientad_model.eval()
    # Move the model to the GPU if you want to use it for inference
    if torch.cuda.is_available():
        loaded_efficientad_model.to("cuda")
        print("Model moved to GPU.")

except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {EFFICIENTAD_CHECKPOINT_PATH}")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")


# --- Example of using the loaded model for inference ---
# (You would typically do this in a separate script or function)
# You would need to load or create a DataModule or prepare individual images
# For a single image prediction:
# from PIL import Image
# import torchvision.transforms as T
# import torch
#
# def predict_single_image(model, image_path, image_size=256):
#     # Load and preprocess the image
#     image = Image.open(image_path).convert("RGB")
#     # Use the same transforms as during training/evaluation (resize, normalize)
#     transform = T.Compose([
#         T.Resize((image_size, image_size)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
#     ])
#     processed_image = transform(image).unsqueeze(0) # Add batch dimension
#
#     # Move image to the same device as the model
#     device = next(model.parameters()).device
#     processed_image = processed_image.to(device)
#
#     # Perform inference (forward pass)
#     with torch.no_grad(): # No gradient calculation needed for inference
#         # The exact method to get anomaly scores depends on the model's implementation
#         # Anomalib models typically have a forward or predict method that returns scores/maps
#         output = model(processed_image)
#
#     # Process the output (this is model-dependent)
#     # For Ganomaly, output might be a tuple/dict containing anomaly score and map
#     # You'd need to refer to the Ganomaly model's code/docs to know the exact output format
#     print("Inference output (model-dependent):", output)
#     # You'd typically normalize scores, apply thresholds, and visualize here.
#
# # Example usage:
# # if loaded_ganomaly_model:
# #     predict_single_image(loaded_ganomaly_model, "path/to/a/new/test_image.jpg", image_size=256)


# You can similarly load EfficientAd:
# from anomalib.models.image.efficientad import EfficientAd
# EFFICIENTAD_CHECKPOINT_PATH = "results/EfficientAd/Folder/last.ckpt" # Example path
# try:
#     loaded_efficientad_model = EfficientAd.load_from_checkpoint(
#         EFFICIENTAD_CHECKPOINT_PATH,
#         # EfficientAd specific args might be needed here if not saved in checkpoint,
#         # e.g., image_size=256, model_size="S"
#     )
#     print(f"Successfully loaded EfficientAd model from {EFFICIENTAD_CHECKPOINT_PATH}")
#     loaded_efficientad_model.eval()
#     if torch.cuda.is_available():
#          loaded_efficientad_model.to("cuda")
#          print("EfficientAd model moved to GPU.")
# except FileNotFoundError:
#     print(f"Error: Checkpoint file not found at {EFFICIENTAD_CHECKPOINT_PATH}")
# except Exception as e:
#     print(f"An error occurred while loading the EfficientAd model: {e}")