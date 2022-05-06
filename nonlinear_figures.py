import torch
import matplotlib.pyplot as plt
import numpy as np

X = torch.load("data/X.pt", map_location=torch.device("cpu"))
y = torch.load("data/y.pt", map_location=torch.device("cpu")).reshape(-1)

W = torch.load("data/best_W.pt", map_location=torch.device("cpu"))
net = torch.load("data/best_net.pt", map_location=torch.device("cpu"))

U_W = (W.T @ X).reshape(-1)
U_net = (net.forward(X.T).T).reshape(-1).detach()

plt.hist(U_W[y == 1].numpy(), color='r', alpha=0.5, bins=np.linspace(-7.5, 10, num=50))
plt.hist(U_W[y == 0].numpy(), color='b', alpha=0.5, bins=np.linspace(-7.5, 10, num=50))
plt.title("Distribution in Linearly Transformed Space")
plt.savefig("figures/linear.png")

plt.close()

plt.hist(U_net[y == 1].numpy(), color='r', alpha=0.5, bins=np.linspace(-40, 40, num=50))
plt.hist(U_net[y == 0].numpy(), color='b', alpha=0.5, bins=np.linspace(-40, 40, num=50))
plt.title("Distribution in Neural Network Transformed Space")
plt.savefig("figures/nonlinear.png")