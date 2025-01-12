import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
import os
from models.MLP import MLP

# 图像归一化函数
def normalize_image(im_flatten):
    max_val = torch.max(im_flatten)
    min_val = torch.min(im_flatten)
    im_flatten = (im_flatten - min_val) / (max_val - min_val + 1e-8)
    return im_flatten

def perform_attack(target_label):
    model_dir = "./data/target/mynet_50.pkl"
    attack_dir = "./data/attack/"
    h, w = 112, 92
    alpha = 5000
    learning_rate = 0.1
    class_num = 40

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载目标模型
    assert os.path.exists(model_dir), "模型文件不存在！"
    model = MLP(h * w, class_num).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    # 初始化攻击图像
    aim_flatten = torch.zeros(1, h * w).to(device).requires_grad_()
    optimizer = torch.optim.SGD([aim_flatten], lr=learning_rate)

    # 攻击过程
    for _ in tqdm(range(alpha), desc="Performing Attack"):
        optimizer.zero_grad()
        # 前向传播
        output = model(aim_flatten)
        target = torch.tensor([target_label]).to(device)
        loss = F.cross_entropy(output, target)

        # 反向传播与优化
        loss.backward()
        optimizer.step()
         # 归一化与裁剪
        aim_flatten.data = normalize_image(aim_flatten.data)
        aim_flatten.data = torch.clamp(aim_flatten.data, 0, 1)

    # 保存攻击结果
    os.makedirs(attack_dir, exist_ok=True)
    result_image_path = os.path.join(attack_dir, f"inverted_{target_label}.png")
    save_image(aim_flatten.view(1, 1, h, w), result_image_path)
    print(f"攻击完成，结果已保存至: {result_image_path}")
    return result_image_path
