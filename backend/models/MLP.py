import torch
import torch.nn as nn


# 模型定义
class MLP(nn.Module):
    def __init__(self, input_features, output_features):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_features, 3000)
        self.regression = nn.Linear(3000, output_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.regression(x)
        return x