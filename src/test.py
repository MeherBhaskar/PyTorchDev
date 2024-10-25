import torch

if torch.cuda.is_available():
    print("GPU is available")
    num_gpu = torch.cuda.device_count()
    print("Number of GPUs:", num_gpu)
    for i in range(num_gpu):
        print("GPU", i, ":", torch.cuda.get_device_name(i))
else:
    print("GPU is not available")