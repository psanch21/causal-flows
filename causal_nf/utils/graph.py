import torch


def matrix_power(matrix, n):
    result = torch.eye(matrix.size(0))
    for _ in range(n):
        result = torch.matmul(result, matrix)
    return result


def ancestor_matrix(adjacency_matrix):
    n_vertices = adjacency_matrix.size(0)
    matrix_power_ = matrix_power(adjacency_matrix, n_vertices - 1)

    ancestor_matrix = (matrix_power_ > 0).float()
    return ancestor_matrix
