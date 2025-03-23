import torch
import torch.nn as nn

MODEL_CLASS_NAME = "MLP"
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
    
    def predict(self, x):
        """ 只返回置信度向量 """
        self.eval()  # 评估模式
        with torch.no_grad():
            logits = self.forward(x)  # 计算输出
            output = F.softmax(logits, dim=-1)  # 归一化
        return output  # 仅返回置信度