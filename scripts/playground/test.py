# Test script to show the use of BCE loss for multi-label classification
# Taken from here:
import torch
from torch import nn, optim

model = nn.Linear(20, 5)  # predict logits for 5 classes
x = torch.randn(2, 20)
y = torch.tensor(
    [[1.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]]
)  # get classA and classC as active

criterion = nn.BCEWithLogitsLoss(reduction="none")
criterion_sum = nn.BCEWithLogitsLoss(reduction="sum")
criterion_mean = nn.BCEWithLogitsLoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)  # (2, 5)
    loss_sum = criterion_sum(output, y)  # Sums loss from entire loss tensor
    loss_mean = criterion_mean(output, y)  # Averages loss from entire loss tensor
    loss.backward()
    optimizer.step()
    print("Loss: {:.3f}".format(loss.item()))
    print("Loss w/ reduction sum: {:.3f}".format(loss_sum.item()))
    print("Loss w/ reduction mean: {:.3f}".format(loss_mean.item()))
