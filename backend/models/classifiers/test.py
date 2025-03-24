# 测试模型文件测试接口的统一性
from target_vgg16 import VGG16
from target_mlp import MLP
import torch
from PIL import Image
from torchvision import transforms
# 图像加载和预处理
def image_to_tensor(image_path): #得想办法图像大小，CNN是每次除以5次
    image = Image.open(image_path)

    # 获取图像的宽度和高度
    width, height = image.size

    print(f"图像尺寸: 宽度 {width} 像素, 高度 {height} 像素")

    # 图像预处理
    if image.mode == "L":
        print("图像是灰度图（Grayscale）")
        image = image.convert("L")  # 确保转换为灰度
    elif image.mode in ["RGB", "RGBA"]:
        print("图像是彩色图（Color）")
        image = image.convert("RGB")  # 转换为标准RGB格式
    else:
        print(f"未知的图像模式: {image.mode}")

    transform = transforms.Compose([
        transforms.Resize(image.size),  # 调整图像大小
        transforms.ToTensor(),  # 转为 Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    ])
    return transform(image).unsqueeze(0)  # 扩展维度为 (1, (原来的))

# 加载模型参数
def load_model(model_class, model_path, device, class_num):
    model = model_class(class_num).to(device) 
    if model_path[-3:] == "tar":                #要加个文件类型判断
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device)) 
    model.eval()
    return model

# 主程序
if __name__ == "__main__":
    model_path = "MLP.pkl"  # 模型参数文件路径
    image_path = "test_MLP_ATT1.png"  # 输入图像路径
    class_num = 40  # 假设分类数为40

    # model_path = "VGG16_88.26.tar"  # 模型参数文件路径
    # image_path = "16.png"  # 输入图像路径
    # class_num = 1000  # 假设分类数为10


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备

    # 1. 加载模型
    mlp_model = load_model(MLP, model_path, device, class_num)
    # vgg16_model = load_model(VGG16, model_path, device, class_num)


    # 2. 载入并处理图像
    image_tensor = image_to_tensor(image_path).to(device)

    # 3. 使用 MLP 模型进行预测
    output = mlp_model.predict(image_tensor)

    # 4. 输出预测结果
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"预测类别: {predicted_class}")
    print(f"预测概率: {output[0]}")