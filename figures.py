import torch
import matplotlib.pyplot as plt
import numpy as np

full_dim_beta_errors = np.array(torch.load("full_dim_errors.pt"))
low_dim_beta_errors = np.array(torch.load("low_dim_transformed_errors.pt"))

full_dim_test_losses = np.array(torch.load("test_losses.pt"))
low_dim_test_losses = np.array(torch.load("low_dim_test_losses.pt"))

fig = plt.figure()
plt.plot(np.arange(1, 11), np.median(full_dim_beta_errors, axis=0), c='r', label='Full dimension')
plt.fill_between(np.arange(1, 11), np.percentile(full_dim_beta_errors, 25, axis=0), np.percentile(full_dim_beta_errors, 75, axis=0), facecolor='r', alpha=0.5)

plt.plot(np.arange(1, 11), np.median(low_dim_beta_errors, axis=0), c='b', label='Reduced dimension')
plt.fill_between(np.arange(1, 11), np.percentile(low_dim_beta_errors, 25, axis=0), np.percentile(low_dim_beta_errors, 75, axis=0), facecolor='b', alpha=0.5)
plt.yscale("log")
plt.legend()
plt.xlabel("Input dimension")
plt.ylabel("Beta estimation squared error")
plt.savefig("beta_errors.png")
plt.close(fig)

fig = plt.figure()
plt.plot(np.arange(1, 11), np.median(full_dim_test_losses, axis=0), c='r', label='Full dimension')
plt.fill_between(np.arange(1, 11), np.percentile(full_dim_test_losses, 25, axis=0), np.percentile(full_dim_test_losses, 75, axis=0), facecolor='r', alpha=0.5)

plt.plot(np.arange(1, 11), np.median(low_dim_test_losses, axis=0), c='b', label='Reduced dimension')
plt.fill_between(np.arange(1, 11), np.percentile(low_dim_test_losses, 25, axis=0), np.percentile(low_dim_test_losses, 75, axis=0), facecolor='b', alpha=0.5)
plt.yscale("log")
plt.legend()
plt.xlabel("Input dimension")
plt.ylabel("Test set regression mean squared error")
plt.savefig("mse.png")
plt.close(fig)
