import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

MUTATION_RATE = 0.2
LR = 0.1


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.linear2 = nn.Linear(8, output_size)
        self.fitness = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def mutate(self):
        for layer in [self.linear1, self.linear2]:
            weights, biases = layer.weight.data, layer.bias.data
            weights, biases = weights.detach().numpy(), biases.detach().numpy()
            weights_noise = np.random.normal(size=weights.shape) * np.random.binomial(1, MUTATION_RATE,
                                                                                      size=weights.shape)
            biases_noise = np.random.normal(size=biases.shape) * np.random.binomial(1, MUTATION_RATE, size=biases.shape)
            np.random.binomial(1, MUTATION_RATE, size=None)
            weights += LR * weights_noise
            biases += LR * biases_noise
            with torch.no_grad():
                layer.weight.data = torch.Tensor(weights)
                layer.bias.data = torch.Tensor(biases)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self,mode=None):
        if mode is None:
            self.load_state_dict(torch.load(f'./model/model.pth'))
        else:
            self.load_state_dict(torch.load(f'./model/model_{mode}.pth'))
