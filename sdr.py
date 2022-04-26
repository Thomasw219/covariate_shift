import torch
import numpy as np

from toy_data import generate_toy_data
from kmmClassification import kmm

def gaussian_kernel_gram_matrix(X, sigma=0.1, l=1):
    n = X.shape[1]
    difference_matrix = torch.stack([X - X[:, i:i + 1] for i in range(n)])
    norm_matrix = torch.norm(difference_matrix, dim=1)
    gram_matrix = sigma**2 * torch.exp(-1 * norm_matrix / (2 * l**2))
    return gram_matrix

def centralized_gram_matrix(X):
    n = X.shape[1]
    d = X.shape[0]
    gram_matrix = gaussian_kernel_gram_matrix(X)
    center = torch.eye(n, device=X.device) - 1 / n * torch.ones(n, n)
    return center @ gram_matrix @ center

def conditional_covariance_matrix(K_cy, K_cy2, K_cu, epsilon=0.01):
    n = K_cy.shape[0]
    identity = torch.eye(n, device=K_cy.device)
    inner = torch.inverse(K_cu + epsilon * identity)
    outer = K_cy @ K_cu
    return K_cy2 - outer @ inner @ inner @ outer.T

if __name__ == "__main__":
    n_samples = 10
    all_data = []
    device = torch.device("cuda:0")
    for sample in range(n_samples):
        data = []
        for d in range(1, 21):
            n = 500
            X_tr, y_tr, X_te, y_te, betas_true = generate_toy_data(d, n, n, device=device)
            K_cy = centralized_gram_matrix(y_tr)
            identity = torch.eye(n, device=K_cy.device)
            K_cy2 = (K_cy + 0.01 * identity) @ (K_cy + 0.01 * identity)

            d_r = 1
            iterations = 500
            eta = 0.5
            W = torch.ones(d, d_r)
            torch.nn.init.orthogonal_(W)
            for i in range(iterations):
                W = W / torch.norm(W.flatten())
                check = W.T @ W
                check[check < 1e-3] = 0
                W = W.detach()
                W.requires_grad = True

                U_tr = W.T @ X_tr
                K_cu = centralized_gram_matrix(U_tr)
                objective = torch.trace(conditional_covariance_matrix(K_cy, K_cy2, K_cu))
                print(objective.item())

                print(W.T)
                W.grad = None
                objective.backward()
                W = W.detach() - eta * W.grad
                if np.abs(np.abs(W[0, 0]) - 1) < 0.025:
                    break

            W = W / torch.norm(W.flatten())
            U_tr = W.T @ X_tr
            U_te = W.T @ X_te

            betas = torch.tensor(kmm(U_tr.cpu().numpy().T.astype(np.double), U_te.cpu().numpy().T.astype(np.double), 0.01))
            data.append(torch.sum(torch.square(betas_true - betas)).item())
        print(data)
        all_data.append(data)
    print(torch.tensor(all_data))
    torch.save(all_data, "low_dim_errors.pt")
