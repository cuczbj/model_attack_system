# restruct.py
import os
from attack import get_attack_method
import torch
from models.MLP import MLP

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
2.目标标签
未完成
3.目标模型文件目录（或者直接文件路径吧）
4.迭代数
5.学习率
"""
def reconstruct(attack_method_name, target_label):
    """执行指定的攻击方法"""
    attack_method = get_attack_method(attack_method_name)
    return attack_method(target_label)

# if __name__ == "__main__":
#     # print(reconstruct("standard_attack",12,))
#     print(reconstruct("PIG_attack",12,))


