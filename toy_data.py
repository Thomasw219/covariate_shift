import torch

def generate_toy_data(dim, n_tr, n_te, device=torch.device("cpu")):
    X_tr = torch.normal(torch.zeros(dim, n_tr, device=device), torch.ones(dim, n_tr, device=device))
    y_tr = torch.sigmoid(X_tr[0:1, :] + 5 * torch.tan(X_tr[0:1, :]))

    X_te = torch.normal(0.5 * torch.ones(dim, n_te, device=device), torch.ones(dim, n_te, device=device))
    y_te = torch.sigmoid(X_te[0:1, :] + 5 * torch.tan(X_te[0:1, :]))

    return X_tr, y_tr, X_te, y_te
