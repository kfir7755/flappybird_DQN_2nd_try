import torch
import torch.nn as nn
import torch.nn.functional as F
import os

MUTATION_RATE = 0.1
LR = 0.1


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size,device):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.linear2 = nn.Linear(16, output_size)
        self.fitness = 0
        self.to(device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def mutate(self,lr=LR):
        for layer in [self.linear1, self.linear2]:
            weights, biases = layer.weight.data, layer.bias.data
            weights_noise = torch.normal(1,0,size=weights.shape) * torch.bernoulli(torch.ones(weights.shape) * MUTATION_RATE)
            biases_noise = torch.normal(1,0,size=biases.shape) * torch.bernoulli(torch.ones(biases.shape) * MUTATION_RATE)
            weights += lr * weights_noise
            biases += lr * biases_noise
            with torch.no_grad():
                layer.weight.data = weights
                layer.bias.data = biases

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
