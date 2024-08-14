import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDNN version:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("NCCL version:", torch.cuda.nccl.version())