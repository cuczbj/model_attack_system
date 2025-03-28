# restruct.py
import os
from attack import get_attack_method
import torch
# from models.classifiers.target_mlp import MLP
from PIL import Image
from upload_importlib import load_model

# def reload_model(model_path, h, w, class_num=40):
#     """重载模型"""
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
#     model = MLP(h * w, class_num)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     print(f"模型成功加载: {model_path}")
#     return model

# def reconstruct(attack_method_name, target_label, model, h, w, alpha=5000, learning_rate=0.1):
#     """执行指定的攻击方法"""
#     attack_method = get_attack_method(attack_method_name)
#     return attack_method(target_label, model, h, w, alpha, learning_rate)

"""
1攻击方法
2.目标模型（已加载）
3.目标标签
未完成
4.目标模型文件目录（或者直接文件路径吧）
5.迭代数
6.学习率
"""
def reconstruct(attack_method_name, model, target_label,  h, w, device, task_id=None):
    """执行指定的攻击方法，并使用任务ID标识结果"""
    attack_method = get_attack_method(attack_method_name)

    # 加载目标模型


    try:
        # 打印调试信息
        print(f"执行攻击方法: {attack_method_name}, 目标标签: {target_label}, 任务ID: {task_id}")
        # 传递任务ID给攻击方法
        result = attack_method(target_label, model,  h, w, device, task_id)
        return result
    except Exception as e:
        print(f"攻击执行出错: {e}")
        raise e

# if __name__ == "__main__":
#     # print(reconstruct("standard_attack",12,))
#     print(reconstruct("PIG_attack",12,))


