import torch

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

print("CUDA available:", torch.cuda.is_available())
print("Current device index:", torch.cuda.current_device())
print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
