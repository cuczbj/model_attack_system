# attack/__init__.py

from .standard_attack import standard_attack
from .new_attack import new_attack

# 可选：将攻击方法映射到一个字典中，方便调用
ATTACK_METHODS = {
    "standard_attack": standard_attack,
    "new_attack": new_attack,
}

def get_attack_method(method_name):
    """根据攻击方法名获取攻击方法"""
    attack_method = ATTACK_METHODS.get(method_name)
    if not attack_method:
        raise ValueError(f"未知的攻击方法: {method_name}")
    return attack_method