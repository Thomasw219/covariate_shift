from re import X
import torch
import numpy as np

import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from toy_data import generate_toy_data, transformed_betas, generate_toy_data_nonlinear, transformed_betas_nonlinear, non_linear_data
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
    center = torch.eye(n, device=X.device) - 1 / n * torch.ones(n, n, device=X.device)
    return center @ gram_matrix @ center

def conditional_covariance_matrix(K_cy, K_cy2, K_cu, epsilon=0.01):
    n = K_cy.shape[0]
    identity = torch.eye(n, device=K_cy.device)
    inner = torch.inverse(K_cu + epsilon * identity)
    outer = K_cy @ K_cu
    return K_cy2 - outer @ inner @ inner @ outer.T

def proof_of_concept_experiment():
    n_samples = 10
    all_full_dim_errors = []
    all_errors = []
    all_transformed_errors = []
    all_test_losses = []
    all_transformed_test_losses = []
    device = torch.device("cuda:0")
    for sample in range(n_samples):
        full_dim_errors = []
        errors = []
        transformed_errors = []
        test_losses = []
        transformed_test_losses = []
        for d in range(1, 11):
            n = 500
            X_tr, y_tr, X_te, y_te, betas_true = generate_toy_data(d, n, n, device=device)
            K_cy = centralized_gram_matrix(y_tr)
            identity = torch.eye(n, device=K_cy.device)
            K_cy2 = (K_cy + 0.01 * identity) @ (K_cy + 0.01 * identity)

            d_r = 1
            iterations = 500
            eta = 0.25
            W = torch.ones(d, d_r, device=device)
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
                if np.abs(np.abs(W[0, 0].item()) - 1) < 0.025:
                    break
                W.grad = None
                objective.backward()
                W = W.detach() - eta * W.grad
                if i % 100 == 0 and i != 0:
                    W = torch.ones(d, d_r, device=device)

            W = W.detach()
            W = W / torch.norm(W.flatten())
            U_tr = W.T @ X_tr
            U_te = W.T @ X_te

            full_dim_betas = torch.tensor(kmm(X_tr.cpu().numpy().T.astype(np.double), X_te.cpu().numpy().T.astype(np.double), 0.01), device=device)
            betas = torch.tensor(kmm(U_tr.cpu().numpy().T.astype(np.double), U_te.cpu().numpy().T.astype(np.double), 0.01), device=device)
            betas_transformed_true = transformed_betas(W, U_tr, device=device)
            full_dim_errors.append(torch.sum(torch.square(betas_true - full_dim_betas)).item())
            errors.append(torch.sum(torch.square(betas_true - betas)).item())
            transformed_errors.append(torch.sum(torch.square(betas_transformed_true - betas)).item())

            model = LinearRegression()
            model.fit(X_tr.T.cpu().numpy(), y_tr.flatten().cpu().numpy(), full_dim_betas.cpu().numpy())
            y_te_hat = model.predict(X_te.T.cpu().numpy())
            test_losses.append(mean_squared_error(y_te.flatten().cpu().numpy(), y_te_hat))
            model.fit(U_tr.T.cpu().numpy(), y_tr.flatten().cpu().numpy(), betas.cpu().numpy())
            y_te_hat = model.predict(U_te.T.cpu().numpy())
            transformed_test_losses.append(mean_squared_error(y_te.flatten().cpu().numpy(), y_te_hat))
        all_full_dim_errors.append(full_dim_errors)
        all_errors.append(errors)
        all_transformed_errors.append(transformed_errors)
        all_test_losses.append(test_losses)
        all_transformed_test_losses.append(transformed_test_losses)

    torch.save(all_full_dim_errors, "full_dim_errors.pt")
    torch.save(all_errors, "low_dim_errors.pt")
    torch.save(all_transformed_errors, "low_dim_transformed_errors.pt")
    torch.save(all_test_losses, "test_losses.pt")
    torch.save(all_transformed_test_losses, "low_dim_test_losses.pt")

def nonlinear_experiment():
    d = 2
    d_r = 1
    iterations = 500
    eta = 0.25
    n = 400
    device = torch.device("cuda:0")
    X, y = non_linear_data(n, device)

    plt.scatter(X.T[y == 1][:, 0].cpu(), X.T[y == 1][:, 1].cpu(), c='r')
    plt.scatter(X.T[y == 0][:, 0].cpu(), X.T[y == 0][:, 1].cpu(), c='b')
    plt.savefig("figures/nonlinear_2d.png")

    y = y.reshape(1, n)
#    torch.save(X, "data/X.pt")
#    torch.save(y, "data/y.pt")

    print(torch.sum(y == 1))
    print(torch.sum(y == 0))

    best_obj = torch.inf
    best_W = None
    for i in range(10):
        W = torch.ones(d, d_r, device=device)
        torch.nn.init.orthogonal_(W)

        K_cy = centralized_gram_matrix(y)
        identity = torch.eye(n, device=K_cy.device)
        K_cy2 = (K_cy + 0.01 * identity) @ (K_cy + 0.01 * identity)

        for i in range(iterations):
            W = W / torch.norm(W.flatten())
            check = W.T @ W
            check[check < 1e-3] = 0
            W = W.detach()
            W.requires_grad = True

            U_tr = W.T @ X
            K_cu = centralized_gram_matrix(U_tr)
            objective = torch.trace(conditional_covariance_matrix(K_cy, K_cy2, K_cu))
            print(objective.item())

            print(W.T)
            if np.abs(np.abs(W[0, 0].item()) - 1) < 0.025:
                break
            W.grad = None
            objective.backward()
            W = W.detach() - eta * W.grad
            if i % 100 == 0 and i != 0:
                W = torch.ones(d, d_r, device=device)
            print(objective.item())

        W = W.detach()
        W = W / torch.norm(W.flatten())
        if objective < best_obj:
            best_obj = objective
            best_W = W
    print(best_obj)
    exit()
#    print(best_W)
#    torch.save(best_W, "data/best_W.pt")

    best_obj = torch.inf
    best_net = None
    for i in range(10):
        print("New net")
        net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        K_cy = centralized_gram_matrix(y)
        identity = torch.eye(n, device=K_cy.device)
        K_cy2 = (K_cy + 0.01 * identity) @ (K_cy + 0.01 * identity)

        for i in range(iterations):
            U_tr = net.forward(X.T).T
            K_cu = centralized_gram_matrix(U_tr)
            objective = torch.trace(conditional_covariance_matrix(K_cy, K_cy2, K_cu))
            print(objective.item())

            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

        if objective < best_obj:
            best_obj = objective
            best_net = net
    print(best_obj)
    print(best_net)
    torch.save(best_net, "data/best_net.pt")

if __name__ == "__main__":
    nonlinear_experiment()
