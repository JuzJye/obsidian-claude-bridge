#!/usr/bin/env python3
"""CPU autograd cross-check for Day 7's analytical least-squares example."""
import json
import torch

torch.set_num_threads(1)
torch.manual_seed(7)
dtype = torch.float64
x = torch.tensor([[1., 0.], [1., 1.], [1., 2.]], dtype=dtype, device="cpu")
y = torch.tensor([[1.], [2.], [2.]], dtype=dtype, device="cpu")
theta = torch.zeros((2, 1), dtype=dtype, device="cpu", requires_grad=True)
prediction = x @ theta
assert prediction.shape == y.shape == (3, 1)
loss = 0.5 * (prediction - y).square().mean()
loss.backward()
initial_gradient = theta.grad.detach().clone()
expected_gradient = torch.tensor([[-5 / 3], [-2.]], dtype=dtype)
assert torch.allclose(initial_gradient, expected_gradient)

optimizer = torch.optim.SGD([theta], lr=0.2)
for _ in range(500):
    optimizer.zero_grad(set_to_none=True)
    prediction = x @ theta
    loss = 0.5 * (prediction - y).square().mean()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    reference = torch.linalg.lstsq(x, y).solution
    final_loss = 0.5 * (x @ theta - y).square().mean()
    assert torch.allclose(theta, reference, atol=1e-8, rtol=1e-8)
print(json.dumps({"torch": torch.__version__, "device": str(theta.device),
                  "dtype": str(dtype), "seed": 7, "iterations": 500,
                  "initial_gradient": initial_gradient.flatten().tolist(),
                  "fitted_beta": theta.detach().flatten().tolist(),
                  "lstsq_beta": reference.flatten().tolist(),
                  "final_loss": final_loss.item(), "checks": "passed"}, indent=2))
