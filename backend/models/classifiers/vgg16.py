## vgg16_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

MODEL_CLASS_NAME ="VGG16"

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        # x = x.unsqueeze(0)  # 增加 batch 维度
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        logits = self.fc_layer(feature)
        return feature, logits  # 统一返回 (特征, 逻辑回归值)

    def predict(self, x):
        """ 直接返回 softmax 结果 """
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(x)
            return F.softmax(logits, dim=1)  # 归一化概
    

