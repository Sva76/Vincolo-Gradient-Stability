import torch
from torch import nn, optim
from vincolo import VincoloController

# Dummy model
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

vincolo = VincoloController()

# Dummy data
for step in range(20):
    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()

    vincolo.step(model)  # Vincolo acts here

    optimizer.step()

    print(f"step={step} loss={loss.item():.4f}")
