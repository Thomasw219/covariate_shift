import torch

from toy_data import generate_toy_data

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
#    print(gram_matrix)
    return center @ gram_matrix @ center

def conditional_covariance_matrix(K_cy, K_cy2, K_cu, epsilon=0.01):
    n = K_cy.shape[0]
    identity = torch.eye(n, device=K_cy.device)
    inner = torch.inverse(K_cu + epsilon * identity)
    outer = K_cy @ K_cu
#    print(torch.trace(K_cy2), torch.trace(outer @ inner @ inner @ outer.T))
    return K_cy2 - outer @ inner @ inner @ outer.T

if __name__ == "__main__":
    n = 100
    d = 50
    X_tr, y_tr, X_te, y_te = generate_toy_data(5, n, n)
    K_cy = centralized_gram_matrix(y_tr)
    K_cx = centralized_gram_matrix(X_tr)
    identity = torch.eye(n, device=K_cy.device)
    K_cy2 = (K_cy + 0.01 * identity) @ (K_cy + 0.01 * identity)
    objective_no_reduction = torch.trace(conditional_covariance_matrix(K_cy, K_cy2, K_cx))
    print(objective_no_reduction)

    K_cu = centralized_gram_matrix(X_tr[1:, :])
    objective_wrong_reduction = torch.trace(conditional_covariance_matrix(K_cy, K_cy2, K_cu))
    print(objective_wrong_reduction)

    K_cu = centralized_gram_matrix(X_tr[0:1, :])
    objective_full_reduction = torch.trace(conditional_covariance_matrix(K_cy, K_cy2, K_cu))
    print(objective_full_reduction)

    print(objective_full_reduction < objective_wrong_reduction)
    print(objective_no_reduction < objective_wrong_reduction)
    print(objective_full_reduction < objective_no_reduction)
