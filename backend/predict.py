import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
# from models.classifiers.target_mlp import MLP
# from models.classifiers import VGG16, IR152, FaceNet, FaceNet64
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from upload_importlib import load_model
from typing import Any, Tuple



# def load_model(model_name,  h, w, class_num, device):
    
#     if model_name == "VGG16":
#         model = VGG16(class_num)
#         model_path = './checkpoint/target_model/VGG16_88.26.tar'
#     elif model_name == "IR152":
#         model = IR152(class_num)
#         model_path = './checkpoint/target_model/IR152_91.16.tar'
#     elif model_name == "FaceNet64":
#         model = FaceNet64(class_num)
#         model_path = './checkpoint/target_model/FaceNet64_88.50.tar'
#     elif model_name == "MLP":
#         model = MLP(h * w, class_num).to(device)
#         model_path = './checkpoint/target_model/MLP.pkl'  # MLP 使用指定路径
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")
    
    
        
#     # 加载模型状态字典
#     if model_name in ["VGG16", "IR152", "FaceNet64"]:
#         model = torch.nn.DataParallel(model).to(device)  # VGG16, IR152, FaceNet64 使用 DataParallel

#         # 加载模型的权重
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint['state_dict'], strict=False)
#     else:
#         model.load_state_dict(torch.load(model_path))
    
#     model.eval()  # 设置为评估模式
    
#     return model

# def predict_target_model(image, model_name, h, w, class_num):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_model(model_name, h, w, class_num, device)
    
    
    # transform = Compose([Resize((h, w)), ToTensor()])

    # if model_name in ["VGG16", "FaceNet64", "IR152"]:
    #     image_tensor = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

    #     with torch.no_grad():
    #         feature, output = model(image_tensor)  # 兼容返回 feature 和 logits
    #         confidences = torch.softmax(output, dim=-1).squeeze(0)
    #         prediction = torch.argmax(confidences).item()

    #     return prediction, confidences

    # elif model_name == "MLP":
    #     image_tensor = transform(image).view(1, -1).to(device)

    #     with torch.no_grad():
    #         output = model(image_tensor)
    #         confidences = torch.softmax(output, dim=-1).squeeze(0)
    #         prediction = torch.argmax(confidences).item()

    #     return prediction, confidences

    # else:
    #     raise ValueError(f"Unsupported model: {model_name}")


# 预测函数
def predict_target_model(image, model_name, h, w, class_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型（使用model_scanner或upload_importlib中的函数）
    from upload_importlib import load_model, get_available_params
    
    # 获取模型参数文件
    params = get_available_params()
    param_file = None
    for p in params:
        if model_name in p:
            param_file = p
            break
    
    if not param_file:
        raise ValueError(f"找不到模型{model_name}的参数文件")
    
    model = load_model(model_name, param_file, device, class_num)
    
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'predict'):
            output = model.predict(image_tensor)
        else:
            # 如果模型没有predict方法，尝试直接调用
            output = model(image_tensor)
            if isinstance(output, tuple):
                # 某些模型可能返回(特征, 输出)
                output = output[1]
        
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class, output.squeeze(0).cpu().numpy()

# 训练目标模型,无调用，备用,这里只保留前面第一种MLP目标模型
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



# 测试
from PIL import Image
if __name__ == "__main__":
#     # MLP
#     image_path = './test_MLP_ATT.png'  # 替换为实际图像路径
#     h, w = 112, 92  # 输入图像的大小
#     class_num = 40 #  输出类别数量
#     model_name = "MLP"  # 模型名称

    # VGG16
    image_path = './test_VGG16_celeba.png'  # 替换为实际图像路径
    h, w = 64, 64  # 输入图像的大小
    class_num = 1000 #  输出类别数量
    model_name = "MLP"  # 模型名称

#     # 读取并预处理图像
#    image = Image.open(image_path).convert('L')  # 打开图像并转换为灰度图（MLP和其他不一样）
    image = Image.open(image_path).convert('RGB')  # 打开图像并转换为RGB格式（VGG16对应的彩蛇图像）

    #调用
    prediction, confidences = predict_target_model(image, model_name, h, w, class_num)

    # 输出预测结果
    print(f"Predicted Class: {prediction}")
    print(f"Confidence Scores: {confidences}")
    
#     # 这里输出的是一个单一类别的预测，你也可以打印所有类别的置信度
#     # 如果需要查看预测类别的名字，你可以加载一个 ImageNet 的标签映射列表
