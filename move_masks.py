import os
import shutil

# Path to your dataset masks folder
masks_dir = '/home/pavithrag/UNeXt-pytorch/datasets/BUSI/processed/masks'
target_dir = os.path.join(masks_dir, '0')

# Create the '0' subdirectory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Move all files from masks/ to masks/0/
for filename in os.listdir(masks_dir):
    file_path = os.path.join(masks_dir, filename)

    # Skip if it's the newly created '0' folder itself
    if filename == '0' or not os.path.isfile(file_path):
        continue

    shutil.move(file_path, os.path.join(target_dir, filename))

print("✅ All mask images moved to 'masks/0/' successfully.")
