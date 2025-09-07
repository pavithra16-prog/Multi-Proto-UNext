import torch
import time
from ptflops import get_model_complexity_info  # for FLOPs
from torchvision import transforms
from PIL import Image
import os

# -------------------------------
# 1. Force CPU usage
# -------------------------------
device = torch.device("cpu")

# -------------------------------
# 2. Load your UNeXt-Proto model
# -------------------------------
# Make sure this imports your actual model
# from unext_proto_model import UNeXtProto  # replace with your actual import
from proto_model import UNeXtWithPrototypes

# model = UNeXtWithPrototypes()

model = UNeXtWithPrototypes(
    in_channels=3,
    num_classes=1,
    base_c=32,      
    proto_dim=16
)

model.to(device)
model.eval()  # inference mode

# -------------------------------
# 3. Count number of parameters
# -------------------------------
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params / 1e6:.2f} M")

# -------------------------------
# 4. Compute FLOPs (GFlops)
# -------------------------------
# Assuming input size is (3, 256, 256)
with torch.no_grad():
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=False, print_per_layer_stat=False)
    print(f"FLOPs: {flops / 1e9:.2f} GFlops")

# -------------------------------
# 5. Measure average inference time
# -------------------------------
# Replace this path with folder containing 10 images
image_folder = "/home/gpavithra/AIP/images"  # put your 10 test images here
image_files = sorted(os.listdir(image_folder))[:10]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

inference_times = []

with torch.no_grad():
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()

        inference_times.append((end_time - start_time) * 1000)  # ms

avg_time = sum(inference_times) / len(inference_times)
print(f"Average inference time per image: {avg_time:.2f} ms")