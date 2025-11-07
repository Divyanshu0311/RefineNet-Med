import torch
from torchsummary import summary
from scripts.models import UNet, Refiner

model = UNet(in_channels=3, out_channels=1, base_filters=32)
refiner = Refiner(in_channels=4, out_channels=1, base_filters=16)

num_params_unet = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_params_refiner = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
print(f"UNet params: {num_params_unet/1e6:.2f}M")
print(f"Refiner params: {num_params_refiner/1e6:.2f}M")

print("\nUNet Summary:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, (3, 256, 256), device=str(device))
print("\nRefiner Summary:")
refiner = refiner.to(device)
summary(refiner, (4, 256, 256), device=str(device))


print("Checking Infrence time...")
import torch
import time
from torch.autograd import no_grad

def measure_inference_time(model, device, input_size=(1, 3, 320, 320), runs=100):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    # Warm-up (GPU needs this for accurate timing)
    with no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    start = time.time()

    with no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
        torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / runs
    print(f"Average inference time per image: {avg_time*1000:.3f} ms")
    print(f"FPS: {1/avg_time:.2f}")

# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
measure_inference_time(model, device)
