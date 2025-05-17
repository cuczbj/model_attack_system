import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CLASS_NAME = "MLP"
# 模型定义
class MLP(nn.Module):
    def __init__(self, class_num, image_size=(112, 92), hidden_dim=3000):
        super(MLP, self).__init__()
        self.input_dim = image_size[0] * image_size[1]  # 计算输入维度 = 112 * 92
        self.fc = nn.Linear(self.input_dim, hidden_dim)
        self.regression = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平为 1D 向量
        feature = self.fc(x)
        logits = self.regression(feature)
        return feature, logits

    def predict(self, x):
        """ 直接返回 softmax 结果 """
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(x)
            return F.softmax(logits, dim=-1)  # 归一化概率
        