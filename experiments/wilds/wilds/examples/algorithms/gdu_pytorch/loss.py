import torch
from torch import nn


class LayerLoss(nn.Module):
    def __init__(
            self,
            name,
            device,
            criterion,
            sigma,
            lambda_OLS,
            lambda_orth,
            lambda_sparse,
            orthogonal_loss,
            sparse_coding
    ):
        super(LayerLoss, self).__init__()
        self.name = name
        self.device = device
        self.sigma = sigma
        self.criterion = criterion
        self.lambda_OLS = lambda_OLS
        self.lambda_orth = lambda_orth
        self.lambda_sparse = lambda_sparse
        self.orthogonal_loss = orthogonal_loss
        self.sparse_coding = sparse_coding

    def forward(self, output, target, model):
        feature_vector = model.x_tilde
        domain_bases = {i: model.gdu_layer.gdus[f'GDU_{i}'].domain_basis for i in range(model.num_gdus)}
        weight_matrix = model.gdu_layer.betas

        k_x_V = torch.zeros(feature_vector.size(0), len(domain_bases)).to(self.device)
        K = torch.zeros(len(domain_bases), len(domain_bases)).to(self.device)

        # Base Loss
        if isinstance(self.criterion, NLL):
            base_loss = self.criterion(output[0], target[:, :, 0], output[1])
        else:
            base_loss = self.criterion(output, target)

        # OLS Loss (3 terms)
        feature_vector = torch.unsqueeze(feature_vector, -1)
        ols_term_1 = self.rbf_kernel(feature_vector, feature_vector).mean()

        for domain_num, basis_matrix in domain_bases.items():
            k_x_V[:, domain_num] = self.rbf_kernel(feature_vector, basis_matrix).mean(dim=1)
        ols_term_2_matrix = weight_matrix * k_x_V
        ols_term_2_vector = -2 * ols_term_2_matrix.sum(dim=1)
        ols_term_2 = ols_term_2_vector.mean()

        for domain_num_1, basis_matrix_1 in domain_bases.items():
            for domain_num_2, basis_matrix_2 in domain_bases.items():
                K[domain_num_1, domain_num_2] = self.rbf_kernel(torch.permute(basis_matrix_1, (2, 1, 0)), basis_matrix_2).mean()
        ols_term_3 = (weight_matrix @ K @ weight_matrix.transpose(0, 1)).diagonal().mean()
        ols = ols_term_1 + ols_term_2 + ols_term_3
        loss = base_loss + self.lambda_OLS * ols

        # Orthogonal Loss
        if self.orthogonal_loss:
            u, s, v = torch.linalg.svd(K - K.diag().diag())
            orth = s[0]                                     # largest singular value
            loss = loss + self.lambda_orth * orth

        # Sparse Coding Loss (L1 Norm of betas)
        if self.sparse_coding:
            sparse = weight_matrix.abs().sum(dim=1).mean(dim=0)
            loss = loss + self.lambda_sparse * sparse

        return loss

    def rbf_kernel(self, x, y):                                         # todo: in utils make a class out of this which is initialized in loss init with sigma
        l2_dist = torch.sum(torch.square(x-y), dim=1)
        k_x_y = torch.exp(torch.mul(l2_dist, -1/(2*self.sigma**2)))
        return k_x_y


class NLL(nn.Module):
    def __init__(self):
        super(NLL, self).__init__()
        self.criterion = nn.GaussianNLLLoss()

    def forward(self, output, target):
        return self.criterion(output[0], target[:, :, 0], output[1])
