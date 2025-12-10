import torch
from torch.utils.data import DataLoader
from lookhere import LookHere
from data_prep import ImageNetDataset
from datasets import load_dataset

# -------------------------------
# CONFIG
# -------------------------------
batch_size = 4                         # adjust based on GPU memory
pretrained_file = "LH_180_weights_and_config.pth"  # can also use LH_90 or LH_45
img_size = 224                         # normal image size
use_extrapolation = False               # True to test larger images
minival_fraction = 0.01                 # last 1% of train for minival

# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------
# LOAD MODEL
# -------------------------------
checkpoint = torch.load(pretrained_file, map_location=device)
model = LookHere(device=device, lh_config=checkpoint["config"])
model.load_state_dict(checkpoint["weights"])
model = model.eval()
model.to(device)

# -------------------------------
# OPTIONAL: EXTRAPOLATION
# -------------------------------
if use_extrapolation:
    large_img_size = 1024
    model.set_pos_embed(int(large_img_size / 16))
    img_size = large_img_size
    print(f"Testing extrapolation at image size: {img_size}")

# -------------------------------
# PREPARE MINIVAL DATASET
# -------------------------------
# Load HuggingFace ImageNet dataset
print("Loading HuggingFace ImageNet dataset...")
hf_dataset = load_dataset("imagenet-1k", split=f"train[{int((1-minival_fraction)*100)}%:]")

# Wrap with your ImageNetDataset class
minival_dataset = ImageNetDataset(
    dataset=hf_dataset,
    do_augment=False,
    img_size=img_size
)

minival_loader = DataLoader(
    minival_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"Loaded minival dataset with {len(minival_dataset)} images.")

# -------------------------------
# RUN PREDICTIONS
# -------------------------------
print("Running predictions on minival...")
with torch.no_grad():
    for batch in minival_loader:
        images, labels = batch
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=-1)
        print("Predictions:", preds)
        print("Ground truth:", labels)
        break  # remove break to run all batches

print("Done!")

