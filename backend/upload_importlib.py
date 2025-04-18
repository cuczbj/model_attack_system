import importlib
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.resnet64 import ResNetGenerator

MODEL_DIR = "./models/classifiers"
CHECKPOINT_DIR = "./checkpoint/target_model"

# 获取所有可用模型
def get_available_models():
    """ 获取所有可用的模型（即 `models.classifiers` 目录下的 Python 文件）"""
    available_models = {}

    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".py") and filename != "__init__.py" and filename!="evolve.py":
            module_name = filename[:-3]  # 去掉 `.py`
            try:
                module = importlib.import_module(f'models.classifiers.{module_name}')
                
                # 读取模型类名称
                if hasattr(module, "MODEL_CLASS_NAME"):
                    available_models[module_name] = getattr(module, "MODEL_CLASS_NAME")
                else:
                    print(f"Warning: {module_name} does not define MODEL_CLASS_NAME")
            
            except Exception as e:
                print(f"Error loading {module_name}: {e}")

    return available_models

# 获取所有可用模型的参数
def get_available_params():
    """ 获取所有可用的模型参数文件 """
    return [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith((".pth", ".h5",".tar",".pkl"))]

# 载入目标模型
def load_model(model_name, param_filename, device, class_num):
    try:
        # 动态导入模型文件
        model_module = importlib.import_module(f"models.classifiers.{model_name}")
        
        # 检查 MODEL_CLASS_NAME 是否存在
        if not hasattr(model_module, "MODEL_CLASS_NAME"):
            raise RuntimeError(f"MODEL_CLASS_NAME not found in {model_name}.py")

        model_class_name = getattr(model_module, "MODEL_CLASS_NAME")
        model_class = getattr(model_module, model_class_name, None)

        # 实例化模型
        model = model_class(class_num).to(device) 

        # 使用os.path.join构建路径
        param_file = CHECKPOINT_DIR +"/"+  param_filename

        # 检查文件是否存在并可访问
        if not os.path.isfile(param_file):
            raise FileNotFoundError(f"参数文件不存在: {param_file}")
            
        print(f"尝试加载参数文件: {param_file}")

        if param_file.endswith("tar"):
            checkpoint = torch.load(param_file, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(torch.load(param_file, map_location=device)) 
        model.eval()
        return model
    except Exception as e:
        print(f"加载模型出错详情: {str(e)}")  # 打印详细错误
        raise RuntimeError(f"Error loading model {model_name}: {str(e)}")
    
def load_G(attack_method, target_model, dataset, device, class_num):
    try:
        G_DIR = "./checkpoint/G_model"
        # 安全处理target_model名称
        parts = target_model.split("_")
        target_model_name = parts[1] if len(parts) > 1 else target_model
        
        G_param_filename = f"gen_{attack_method}_{target_model_name}_{dataset}.pth.tar"
        G_param_file = G_DIR + "/" + G_param_filename
        print("G_param_file:", G_param_file)
        
        # 检查文件是否存在
        if not os.path.exists(G_param_file):
            raise FileNotFoundError(f"找不到生成器模型文件: {G_param_file}")
        
        # 先固定为ResNetGenerator（PLG使用的生成器模型）
        G = ResNetGenerator(num_classes=class_num).to(device)  # 实例化生成器模型
        gen_ckpt = torch.load(G_param_file)['model'] 
        G.load_state_dict(gen_ckpt)
        G.eval()  
        return G
    except Exception as e:
        raise RuntimeError(f"Error loading G model {attack_method}: {str(e)}")




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

# 示例：获取所有可用模型
# available_models = get_available_models()
# print("Available models:", available_models)