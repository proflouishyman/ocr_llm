import torch
import bitsandbytes as bnb

# Create a tensor
tensor = torch.randn(10, 10).cuda()

# Apply an 8-bit optimizer
optimizer = bnb.optim.Adam8bit([tensor])

print("bitsandbytes installation is successful and has GPU support.")
