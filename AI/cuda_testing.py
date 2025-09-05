import torch

print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("grad enabled:", torch.is_grad_enabled())

if torch.cuda.is_available():
    # ВАЖНО: requires_grad=True
    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    y = x @ x.t()
    y.sum().backward()
    print("OK: backward on CUDA worked")
else:
    x = torch.randn(1024, 1024, requires_grad=True)
    y = x @ x.t()
    y.sum().backward()
    print("OK: backward on CPU worked")
