import torch


def compute_kernel(x, y, kernel_type="rbf", sigma=None):
    if kernel_type == "rbf":
        if sigma is None:
            sigma = torch.median(torch.pdist(x)) + torch.median(torch.pdist(y))
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-(dist**2) / (2 * sigma**2))
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")


def maximum_mean_discrepancy(x, y, kernel_type="rbf", sigma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples x and y.

    Args:
        x (Tensor): A PyTorch tensor of shape (n_x, d), where n_x is the number of samples in x and d is the dimension.
        y (Tensor): A PyTorch tensor of shape (n_y, d), where n_y is the number of samples in y and d is the dimension.
        kernel_type (str): The type of kernel to use. Currently, only 'rbf' (Radial Basis Function) is supported.
        sigma (float, optional): The bandwidth parameter for the RBF kernel. If None, it will be estimated using the median heuristic.

    Returns:
        float: The MMD value between x and y.
    """
    k_xx = compute_kernel(x, x, kernel_type, sigma)
    k_yy = compute_kernel(y, y, kernel_type, sigma)
    k_xy = compute_kernel(x, y, kernel_type, sigma)

    mmd = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
    return mmd
