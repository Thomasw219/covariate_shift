import torch
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.multivariate_normal import MultivariateNormal

def generate_toy_data(dim, n_tr, n_te, device=torch.device("cpu")):
    X_tr = torch.normal(torch.zeros(dim, n_tr, device=device), 0.5 * torch.ones(dim, n_tr, device=device))
    y_tr = torch.sigmoid(X_tr[0:1, :] + 5 * torch.tan(X_tr[0:1, :]))
    X_tr_dist = MultivariateNormal((torch.zeros(dim, 1, device=device)).flatten(), covariance_matrix=(0.5 * torch.eye(dim, device=device)))
#    X_tr_dist = Independent(Normal(torch.tensor([0], device=device), torch.tensor([0.5], device=device)), 1)

    X_te = torch.normal(0.5 * torch.ones(dim, n_te, device=device), 0.5 * torch.ones(dim, n_te, device=device))
    y_te = torch.sigmoid(X_te[0:1, :] + 5 * torch.tan(X_te[0:1, :]))
    X_te_dist = MultivariateNormal(((0.5 * torch.ones(dim, 1, device=device))).flatten(), covariance_matrix=(0.5 * torch.eye(dim, device=device)))
    X_te_dist = Independent(Normal(torch.tensor([0.5], device=device), torch.tensor([0.5], device=device)), 1)

    beta = torch.exp(X_te_dist.log_prob(X_tr.T)) / torch.exp(X_tr_dist.log_prob(X_tr.T))

    return X_tr, y_tr, X_te, y_te, beta


def transformed_betas(W, U_tr, device=torch.device("cpu")):
    d = W.shape[0]
    U_tr_dist = MultivariateNormal((W.T @ torch.zeros(d, 1, device=device)).flatten(), covariance_matrix=W.T @ (0.5 * torch.eye(d, device=device)) @ W)
    U_te_dist = MultivariateNormal((W.T @ (0.5 * torch.ones(d, 1, device=device))).flatten(), covariance_matrix=W.T @ (0.5 * torch.eye(d, device=device)) @ W)
    beta = torch.exp(U_te_dist.log_prob(U_tr.T)) / torch.exp(U_tr_dist.log_prob(U_tr.T))

    return beta

def generate_toy_data_nonlinear(dim, n_tr, n_te, device=torch.device("cpu")):
    X_tr = torch.normal(-1 * torch.ones(dim, n_tr, device=device), 2 * torch.ones(dim, n_tr, device=device))
    y_tr = torch.bernoulli(torch.sigmoid((torch.sqrt(torch.sum(torch.square(X_tr), axis=0)) - 2) * 5))
    X_tr_dist = MultivariateNormal((-1 * torch.ones(dim, 1, device=device)).flatten(), covariance_matrix=(2 * torch.eye(dim, device=device)))
#    X_tr_dist = Independent(Normal(torch.tensor([0], device=device), torch.tensor([0.5], device=device)), 1)

    X_te = torch.normal(torch.ones(dim, n_te, device=device), 2 * torch.ones(dim, n_te, device=device))
    y_te = torch.bernoulli(torch.sigmoid((torch.sqrt(torch.sum(torch.square(X_te), axis=0)) - 2) * 5))
    X_te_dist = MultivariateNormal(((torch.ones(dim, 1, device=device))).flatten(), covariance_matrix=(2 * torch.eye(dim, device=device)))
#    X_te_dist = Independent(Normal(torch.tensor([0.5], device=device), torch.tensor([0.5], device=device)), 1)

    beta = torch.exp(X_te_dist.log_prob(X_tr.T)) / torch.exp(X_tr_dist.log_prob(X_tr.T))

    return X_tr, y_tr, X_te, y_te, beta

def non_linear_data(n, device=torch.device("cpu")):
    X = torch.normal(torch.zeros(2, n, device=device), 3 * torch.ones(2, n, device=device))
    y = torch.bernoulli(torch.sigmoid((torch.sqrt(torch.sum(torch.square(X), axis=0)) - 3.5) * 5))
    return X, y

def transformed_betas_nonlinear(f, U_tr, device=torch.device("cpu")):
    d = W.shape[0]
    U_tr_dist = MultivariateNormal((W.T @ torch.zeros(d, 1, device=device)).flatten(), covariance_matrix=W.T @ (0.5 * torch.eye(d, device=device)) @ W)
    U_te_dist = MultivariateNormal((W.T @ (0.5 * torch.ones(d, 1, device=device))).flatten(), covariance_matrix=W.T @ (0.5 * torch.eye(d, device=device)) @ W)
    beta = torch.exp(U_te_dist.log_prob(U_tr.T)) / torch.exp(U_tr_dist.log_prob(U_tr.T))

    return beta
