import torch
import time
import csv
from ptflops import get_model_complexity_info  # for FLOPs
from torchvision import transforms
from PIL import Image
import os

# -------------------------------
# User config
# -------------------------------
Ks = [1, 2, 3, 4, 5, 6]              # number of prototypes to test
device = torch.device("cpu")         # keep CPU as in your original snippet
image_folder = "/home/gpavithra/AIP/images"  # folder with test images (up to 10)
IN_CHANNELS = 3
H, W = 256, 256                      # input size for FLOPs and inference

# -------------------------------
# Import your model class
# -------------------------------
# Replace with the correct import if needed
from proto_model import UNeXtWithPrototypes

# -------------------------------
# Helper utilities
# -------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops(model, input_res=(3, 256, 256)):
    try:
        flops, params = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
        return flops
    except Exception as e:
        print("ptflops failed:", e)
        return None

def load_image_paths(folder, max_images=10):
    if not os.path.isdir(folder):
        return []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    return [os.path.join(folder, f) for f in files[:max_images]]

transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor()
])

def prepare_input_from_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # 1 x C x H x W

# -------------------------------
# Main routine
# -------------------------------
results = []
image_paths = load_image_paths(image_folder, max_images=10)
use_random = len(image_paths) == 0
if use_random:
    print(f"No images found in '{image_folder}'. Using random tensors for timing.")
else:
    print(f"Using {len(image_paths)} images from '{image_folder}' for timing.")

for K in Ks:
    print("\n" + "="*60)
    print(f"Testing num_prototypes = {K}")

    # Instantiate model using explicit argument name num_prototypes
    try:
        model = UNeXtWithPrototypes(
            in_channels=3,
            num_classes=1,
            base_c=32,
            proto_dim=16,
            num_prototypes=K
        )
    except TypeError as e:
        print(f"Model constructor does not accept 'num_prototypes' or failed: {e}")
        results.append((K, None, None, None))
        continue
    except Exception as e:
        print(f"Failed to construct model for K={K}: {e}")
        results.append((K, None, None, None))
        continue

    model.to(device)
    model.eval()

    # Count parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params / 1e6:.2f} M")

    # Compute FLOPs
    flops = compute_flops(model, (IN_CHANNELS, H, W))
    if flops is not None:
        print(f"FLOPs: {flops / 1e9:.2f} GFlops")
    else:
        print("FLOPs: N/A")

    # Measure average inference time
    inference_times = []
    with torch.no_grad():
        # warm-up (3 runs)
        if use_random:
            inp = torch.randn(1, IN_CHANNELS, H, W).to(device)
        else:
            inp = prepare_input_from_image(image_paths[0]).to(device)

        for _ in range(3):
            _ = model(inp)

        # timed runs (use images if available else random)
        runs = image_paths if not use_random else [None] * 10
        for p in runs:
            if use_random:
                input_tensor = torch.randn(1, IN_CHANNELS, H, W).to(device)
            else:
                input_tensor = prepare_input_from_image(p).to(device)

            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000.0)  # ms

    if len(inference_times) > 0:
        avg_time = sum(inference_times) / len(inference_times)
        std_time = (sum((t - avg_time)**2 for t in inference_times) / len(inference_times))**0.5
        print(f"Average inference time per image: {avg_time:.2f} ms (+/- {std_time:.2f} ms)")
    else:
        avg_time = None
        std_time = None
        print("No inference timings collected.")

    results.append((K, num_params / 1e6, (flops / 1e9) if flops is not None else None, avg_time))

    # cleanup
    del model
    torch.cuda.empty_cache()

# -------------------------------
# Save results to CSV
# -------------------------------
out_csv = "complexity_results.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "params_M", "flops_GF", "avg_inference_ms"])
    for row in results:
        writer.writerow(row)

print(f"\nSaved results to {out_csv}")
print("Done.")
