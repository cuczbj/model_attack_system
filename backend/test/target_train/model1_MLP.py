import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
# from models.MLP import MLP  # 直接绝对导入  # 仅在训练时需要

# class MLP(nn.Module):
#     def __init__(self, input_features, output_features):
#         super(MLP, self).__init__()
#         self.fc = nn.Linear(input_features, 3000)
#         self.regression = nn.Linear(3000, output_features)

#     def forward(self, x):
#         x = self.fc(x)
#         x = self.regression(x)
#         return x

# model_dir = "../../models/model_MLP.pt"  # 改成 .pt，且存的是整个模型
# 预测接口
def predict_target_model(image):
    model_dir = "../../models/model_MLP.pt"  # 改成 .pt，且存的是整个模型
    h, w = 112, 92

    # 加载整个模型（不需要 MLP 代码）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_dir, map_location=device)
    model.eval()

    # 预处理图像
    transform = Compose([Resize((h, w)), ToTensor()])
    image_tensor = transform(image).view(1, -1).to(device)

    # 模型预测
    with torch.no_grad():
        output = model(image_tensor)
        confidences = torch.softmax(output, dim=-1).squeeze(0)
        prediction = torch.argmax(confidences).item()

    return prediction, confidences

# 训练目标模型（无调用，备用）
def train_target_model():
    # dataset_dir = "./data/AT&T_Faces"
    dataset_dir = "D:\Datasets\AT&T Database of Faces"
    model_dir = "../../models/model_MLP.pt"  # 统一用 .pt
    batch_size = 8
    train_epochs = 50
    class_num = 40
    h, w = 112, 92

    # 数据集加载
    transform = Compose([Grayscale(num_output_channels=1), ToTensor()])
    dataset = ImageFolder(dataset_dir, transform=transform)
    train_ds, _ = random_split(dataset, [len(dataset) * 7 // 10, len(dataset) - len(dataset) * 7 // 10])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(h * w, class_num).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # 模型训练
    model.train()
    for epoch in range(train_epochs):
        for images, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{train_epochs}"):
            images = images.to(device).view(images.size(0), -1)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    # 直接保存整个模型（不只是 state_dict）
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    torch.save(model, model_dir)
    return "Model trained and saved successfully!"



if __name__ == "__main__":
    # train_target_model()
    
    #预测接口
    image_path = "inverted_0.png"
    image = Image.open(image_path).convert("L")  # 转为灰度图
    # 预测
    prediction, confidences = predict_target_model(image)

    # 输出结果
    print(f"Predicted class: {prediction}")
    print(f"Confidence scores: {confidences.cpu().numpy()}")
    # 读取测试图片（假设 test.jpg 存在）
    

   
    
    