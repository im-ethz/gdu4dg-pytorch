import torch.nn as nn

from gdu_pytorch.gdu import GDULayer


class LayerModel(nn.Module):
    def __init__(
            self,
            device,
            task,
            feature_extractor,
            feature_vector_size,
            output_size,
            num_gdus=5,
            domain_dim=10,
            kernel_name='RBF',
            sigma=2,
            similarity_measure_name='CS',
            softness_param=1
    ):
        super().__init__()
        self.device = device
        self.task = task
        self.feature_vector_size = feature_vector_size
        self.output_size = output_size
        self.num_gdus = num_gdus
        self.domain_dim = domain_dim
        self.kernel_name = kernel_name
        self.sigma = sigma
        self.similarity_measure_name = similarity_measure_name
        self.softness_param = softness_param

        self.feature_extractor = feature_extractor
        self.gdu_layer = GDULayer(
            device=self.device,
            task=self.task,
            feature_vector_size=self.feature_vector_size,
            output_size=self.output_size,
            num_gdus=self.num_gdus,
            domain_dim=self.domain_dim,
            kernel_name=self.kernel_name,
            sigma=self.sigma,
            similarity_measure_name=self.similarity_measure_name,
            softness_param=self.softness_param
        )
        self.x_tilde = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):                                   # x: (batch_size, input_len, 1)
        self.x_tilde = self.feature_extractor(x)            # x_tilde: (batch_size, hidden_size)
        weighted_prediction = self.gdu_layer(self.x_tilde)  # weighted_prediction: (batch_size, pred_len, 1/2)
        if self.task == 'classification':
            output = weighted_prediction.squeeze()
            output = self.softmax(output)
        elif self.task == 'probabilistic_forecasting':
            mu = weighted_prediction[:, :, 0]               # mu: (batch_size, pred_len)
            sigma = weighted_prediction[:, :, 1]            # sigma: (batch_size, pred_len)
            output = mu, sigma
        return output
