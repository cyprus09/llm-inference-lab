import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS available: {torch.backends.mps.is_available()}")
print(f"Is MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.rand(1000, 1000, device=device)
    y = torch.rand(1000, 1000, device=device)
    z = x @ y
    print(f"Test matmul on MPS device successful, result shape: {z.shape}")
else:
    print(
        "MPS is not available. Please ensure you have a compatible Apple Silicon device and the correct PyTorch version installed."
    )
