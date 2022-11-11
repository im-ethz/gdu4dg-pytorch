import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(
            self,
            device,
            task,
            feature_extractor,
            feature_vector_size,
            output_size,
            num_gdus=2,
    ):
        super().__init__()
        self.needs_y = False
        self.device = device
        self.task = task
        self.feature_extractor = feature_extractor
        self.feature_vector_size = feature_vector_size
        self.output_size = output_size
        self.num_gdus = num_gdus
        self.learning_machines = nn.ModuleDict({
            f'learning_machine_{i}': LearningMachine(self.device,
                                                     self.feature_vector_size,
                                                     self.output_size,
                                                     task=self.task)
            for i in range(self.num_gdus)
        })
        self.output_dim = 1 if task == 'classification' else 2

    def forward(self, x):
        x_tilde = self.feature_extractor(x)
        y_tildes = torch.zeros(x_tilde.size(0), self.num_gdus, self.output_size, self.output_dim).to(self.device)

        for i in range(self.num_gdus):
            y_tildes[:, i, :, :] = self.learning_machines[f'learning_machine_{i}'](x_tilde)

        prediction = torch.mean(y_tildes, dim=1)
        if self.task == 'classification':
            output = prediction.squeeze(dim = 2)
            #output = self.softmax(output)
        elif self.task == 'probabilistic_forecasting':
            mu = prediction[:, :, 0]               # mu: (batch_size, pred_len)
            sigma = prediction[:, :, 1]            # sigma: (batch_size, pred_len)
            output = mu, sigma
        return output


class LearningMachine(nn.Module):
    def __init__(self, device, feature_vector_size, output_size, task):
        super().__init__()
        self.device = device
        self.task = task

        if self.task == 'classification':
            self.linear = nn.Linear(feature_vector_size, output_size)
            #self.activation = nn.Tanh()
        elif self.task == 'probabilistic_forecasting':
            self.linear_mu = nn.Linear(feature_vector_size, output_size)
            self.linear_sigma = nn.Linear(feature_vector_size, output_size)

    def forward(self, x_tilde):
        if self.task == 'classification':
            x = self.linear(x_tilde)
            #y_tilde = self.activation(x)
            y_tilde = x
            y_tilde = y_tilde.unsqueeze(dim=2)
        elif self.task == 'probabilistic_forecasting':
            mu = self.linear_mu(x_tilde)
            sigma = torch.exp(self.linear_sigma(x_tilde))
            y_tilde = torch.stack([mu, sigma], dim=2)
        return y_tilde
