import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from models.MLP import MLP




# 预测接口
def predict_target_model(image):
    model_dir = "./models/mynet_50.pkl"
    h, w = 112, 92
    class_num = 40

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(h * w, class_num).to(device)
    model.load_state_dict(torch.load(model_dir))
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

    # with torch.no_grad():
    #     output = model(img_flatten)  # 模型输出
    #     probabilities = torch.nn.functional.softmax(output, dim=1)  # 计算概率
    #     confidence, prediction = torch.max(probabilities, dim=1)  # 获取置信度和预测类别
    # return jsonify({
    #     "prediction": prediction.item(),
    #     "confidence": confidence.item(),
    #     "probabilities": probabilities.squeeze().tolist()  # 返回每一类别的概率
    
    # })

# 训练目标模型,无调用，备用
def train_target_model():
    dataset_dir = "./data/AT&T_Faces"
    model_dir = "./models/target_model.pkl"
    batch_size = 8
    train_epochs = 20
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

    # 保存模型
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    torch.save(model.state_dict(), model_dir)
    return "Model trained and saved successfully!"
