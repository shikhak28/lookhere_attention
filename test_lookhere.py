import torch
from torch.utils.data import DataLoader
from lookhere import LookHere
from data_prep import ImageNetDataset
from datasets import load_dataset

# -------------------------------
# CONFIG
# -------------------------------
batch_size = 4         # adjust based on GPU memory
pretrained_file = "LH_180_weights_and_config.pth"  # change to LH_90 or LH_45 if needed
img_size = 224         # image size for normal testing
use_extrapolation = False  # set True to test larger images (e.g., 1024)

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
# PREPARE MINIVAL
# -------------------------------
    minival_dataset = ImageNetDataset(
        dataset=load_dataset("imagenet-1k", split="train[99%:]"),  # last 1% as minival
        do_augment=False,
        img_size=img_size,
    )
    minival_loader = DataLoader(
        minival_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

# -------------------------------
# MAKE PREDICTIONS
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
        break  # remove break to run all minival
print("Done!")

