import torch
import torch.nn as nn


class GDULayer(nn.Module):
    def __init__(
            self,
            device,
            task,
            feature_vector_size,
            output_size,
            num_gdus=2,
            domain_dim=3,
            kernel_name='RBF',
            sigma=0.5,
            similarity_measure_name='MMD',
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
        # initialize M GDUs and learning machines
        self.gdus = nn.ModuleDict({
            f'GDU_{i}': GDU(self.device,
                            i,
                            self.domain_dim,
                            self.feature_vector_size,
                            self.kernel_name,
                            self.sigma,
                            self.similarity_measure_name,
                            self.softness_param)
            for i in range(self.num_gdus)
        })
        self.learning_machines = nn.ModuleDict({
            f'learning_machine_{i}': LearningMachine(self.device,
                                                     self.feature_vector_size,
                                                     self.output_size,
                                                     task=self.task)
            for i in range(self.num_gdus)
        })
        self.kernel_softmax = torch.nn.Softmax(dim=1)
        self.betas = None
        self.output_dim = 1 if task == 'classification' else 2
        self.test_counter = 0

    def forward(self, x_tilde):
        self.betas = torch.zeros(x_tilde.size(0), self.num_gdus).to(self.device)                                # betas: (batch_size, num_gdus)
        y_tildes = torch.zeros(x_tilde.size(0), self.num_gdus, self.output_size, self.output_dim).to(self.device)
        weighted_predictions = torch.zeros(x_tilde.size(0), self.num_gdus, self.output_size, self.output_dim).to(self.device)

        print(self.test_counter)

        if self.test_counter == 20:


            self.num_gdus +=1
            print(self.num_gdus)

            self.gdus.update({f'GDU_{self.num_gdus-1}': GDU(self.device, self.num_gdus,
                                                          self.domain_dim,
                                                          self.feature_vector_size,
                                                          self.kernel_name,
                                                          self.sigma,
                                                          self.similarity_measure_name,
                                                          self.softness_param).to(self.device)})

            self.learning_machines.update({f'learning_machine_{self.num_gdus-1}': LearningMachine(self.device,
                                                                                                self.feature_vector_size,
                                                                                                self.output_size,
                                                                                                task=self.task).to(self.device)})
            self.betas = torch.cat((self.betas, torch.zeros(x_tilde.size(0), 1).to(self.device)), 1)
            y_tildes = torch.zeros(x_tilde.size(0), self.num_gdus, self.output_size, self.output_dim).to(self.device)
            weighted_predictions = torch.zeros(x_tilde.size(0), self.num_gdus, self.output_size, self.output_dim).to(self.device)

        if self.test_counter >=20:
            print(self.learning_machines[f'learning_machine_{0}'].linear.weight[0][0])
            print(self.learning_machines[f'learning_machine_{self.num_gdus-1}'].linear.weight[0][0])
        self.test_counter += 1
        for i in range(self.num_gdus):
            self.betas[:, i] = self.gdus[f'GDU_{i}'](x_tilde)
        if self.similarity_measure_name in ['MMD', 'CS']:
            self.betas = self.kernel_softmax(self.betas)
        betas = self.betas.unsqueeze(2).unsqueeze(3)

        for i in range(self.num_gdus):
            y_tildes[:, i, :, :] = self.learning_machines[f'learning_machine_{i}'](x_tilde)
            weighted_predictions[:, i, :, :] = betas[:, i, :, :].clone() * y_tildes[:, i, :, :].clone()

        prediction = torch.sum(weighted_predictions, dim=1)
        return prediction


class GDU(nn.Module):
    def __init__(
            self,
            device,
            gdu_num,
            domain_dim,
            feature_vector_size,
            kernel_name,
            sigma,
            similarity_measure_name,
            softness_param
    ):
        super().__init__()
        self.device = device
        self.gdu_num = gdu_num
        self.domain_dim = domain_dim
        self.feature_vector_size = feature_vector_size
        self.kernel_name = kernel_name
        self.sigma = sigma
        self.similarity_measure_name = similarity_measure_name
        self.softness_param = softness_param

        domain_basis_tensor = torch.normal(
            mean=torch.mul(torch.ones(self.feature_vector_size, self.domain_dim), self.gdu_num*0.5*(-1)**self.gdu_num),
            std=torch.mul(torch.ones(self.feature_vector_size, self.domain_dim), (self.gdu_num+1)*0.1)
        )
        domain_basis_tensor_batch_compatible = torch.unsqueeze(domain_basis_tensor, 0)
        self.domain_basis = torch.nn.Parameter(domain_basis_tensor_batch_compatible)

    def forward(self, x_tilde):
        if self.similarity_measure_name == 'MMD':
            beta = self.mmd(x_tilde)
        elif self.similarity_measure_name == 'CS':
            beta = self.cs(x_tilde)
        elif self.similarity_measure_name == 'Projected':
            beta = self.projected(x_tilde)
        return beta

    def mmd(self, x_tilde):
        x_tilde = torch.unsqueeze(x_tilde, -1)
        k_x_x = self.rbf_kernel(x_tilde, x_tilde)
        k_x_x = torch.squeeze(k_x_x)
        k_x_V = self.rbf_kernel(x_tilde, self.domain_basis)
        k_V_V = self.rbf_kernel(torch.permute(self.domain_basis, (2, 1, 0)), self.domain_basis)
        beta = k_x_x - 2 * torch.mean(k_x_V, dim=1) + torch.mean(k_V_V)
        return beta

    def cs(self, x_tilde):
        x_tilde = torch.unsqueeze(x_tilde, -1)
        k_x_x = self.rbf_kernel(x_tilde, x_tilde)
        k_x_x = torch.squeeze(k_x_x)
        k_x_V = self.rbf_kernel(x_tilde, self.domain_basis)
        k_V_V = self.rbf_kernel(torch.permute(self.domain_basis, (2, 1, 0)), self.domain_basis)
        beta = torch.mean(k_x_V, dim=1) / (k_x_x.sqrt() * k_V_V.sqrt().mean())
        return beta

    def projected(self, x_tilde):
        x_tilde = torch.unsqueeze(x_tilde, -1)
        k_x_V = self.rbf_kernel(x_tilde, self.domain_basis)
        k_V_V = self.rbf_kernel(torch.permute(self.domain_basis, (2, 1, 0)), self.domain_basis)
        beta = torch.mean(k_x_V, dim=1) / k_V_V.mean()
        return beta

    def rbf_kernel(self, x, y):
        l2_dist = torch.sum(torch.square(x-y), dim=1)
        k_x_y = torch.exp(l2_dist * -1/(2*self.sigma**2))
        return k_x_y


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
